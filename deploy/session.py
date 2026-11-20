"""One deterministic simulation session, independent of viewer and training jobs."""

from dataclasses import asdict
from pathlib import Path
from pickle import UnpicklingError
from types import SimpleNamespace

import numpy as np
import torch

from deploy.comparison import reference_qpos
from deploy.control import Controller, Reference
from deploy.motions import motion_catalog
from deploy.recording import TraceBuffer, write_summary
from deploy.world import World
from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_XML,
)
from mjlab.tasks.g1_adapt.ball import (
  BallPhysics,
  RallyBall,
  RallyBallCfg,
)
from mjlab.tasks.g1_adapt.mvae import file_hash
from mjlab.utils.lab_api.math import quat_apply


class SceneProxy(dict):
  env_origins = torch.zeros(1, 3)


class Session:
  def __init__(
    self,
    artifacts,
    motion,
    mode="hierarchy",
    seed=42,
    scenario="continuous",
    noise=0.005,
    dropout=0.05,
    delay_steps=2,
    random_initial_motion=False,
  ):
    torch.manual_seed(seed)
    self.rng = np.random.default_rng(seed)
    self.artifacts, self.motion = artifacts, Path(motion).resolve()
    self.motion_catalog = motion_catalog(artifacts, self.motion)
    self.random_initial_motion = random_initial_motion and mode in (
      "rally",
      "standing",
      "hierarchy",
    )
    self.initial_motions = [
      p for name, p in self.motion_catalog.items() if name.startswith("物理修正动作")
    ] or [self.motion]
    if self.random_initial_motion:
      self.motion = self.initial_motions[
        int(self.rng.integers(len(self.initial_motions)))
      ]
    self.motion_events = []
    self.segment = -1
    self.tracker_loop, self.tracker_at_end = False, False
    expected = artifacts.provenance.get("asset_validation", {}).get("mjcf_sha256")
    if expected and file_hash(G1_PINGPONG_XML) != expected:
      raise ValueError("Robot asset differs from the selected training run")
    self.world = World()
    self.mode, self.seed, self.scenario = mode, seed, scenario
    self.noise, self.dropout, self.delay_steps = noise, dropout, delay_steps
    self.speed, self.latent = 1.0, np.zeros(32, dtype=np.float32)
    self.records, self.trials = TraceBuffer(), []
    self.fall_events = []
    self.control_fault_events = []
    self.control_fault = ""
    self.error: str | None = None
    self.max_record_steps = 0
    self.record_limit_reached = False
    self.paused = False
    self.controller = Controller(artifacts, self.motion, self.world.default_q, mode)
    physics = BallPhysics(**(self.controller.contract.get("ball_physics", {})))
    self.env = SimpleNamespace(
      scene=SceneProxy(robot=self.world.proxy),
      num_envs=1,
      device="cpu",
      step_dt=0.02,
      cfg=SimpleNamespace(seed=seed),
    )
    cfg = RallyBallCfg(
      resampling_time_range=(1e9, 1e9),
      physics=physics,
      noise=noise,
      dropout=dropout,
      delay_steps=delay_steps,
      prediction_steps=self.controller.contract.get("prediction_steps", 30),
    )
    self.ball = RallyBall(cfg, self.env)
    self.estimator = self.ball.estimator
    self.reset()

  @torch.inference_mode()
  def reset(self, reference_time=0.0):
    if reference_time and self.mode != "tracker":
      raise ValueError("Only Tracker supports seeking the reference motion")
    if not np.isfinite(reference_time) or not (
      0 <= reference_time <= self.controller.reference.duration
    ):
      raise ValueError("Reference time must be within this motion")
    if (
      self.random_initial_motion
      and self.mode in ("rally", "standing", "hierarchy")
      and self.records.total
    ):
      self.motion = self.initial_motions[
        int(self.rng.integers(len(self.initial_motions)))
      ]
      self.controller.reference = Reference(self.motion)
    self.controller.reset(reference_time)
    self.control_fault = ""
    self.world.reset(self.controller.reference)
    self.tracker_at_end = reference_time >= self.controller.reference.duration
    self.segment += 1
    self.motion_events.append(
      {
        "step": self.records.total,
        "segment": self.segment,
        "motion": str(self.motion),
        "reference_time_s": float(reference_time),
        "mode": self.mode,
      }
    )
    self.elapsed, self.episode_start, self.next_feed = 0.0, 0.0, 0.0
    self.active = False
    self.ball_in_flight = self.ball_visible = False
    self.trial_fallen, self.was_fallen = False, False
    self.ball.cfg.active = False
    self.ball.done[:] = True
    self.estimator.reset(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1))
    self.info, self.ball_obs = {}, torch.zeros(1, 101)
    self.future = torch.zeros(1, 30, 3)
    self.paused = False
    self.needs_episode_reset = False
    if self.mode in ("rally", "hierarchy", "ball"):
      self.feed()
    self.info = self.measure(
      torch.tensor(self.world.data.ctrl[self.world.actuators].copy())[None]
    )
    self.last_target = torch.tensor(self.info["targets"], dtype=torch.float32)[None]
    self.last_reference_qpos = self.info["reference_qpos"]

  @torch.inference_mode()
  def feed(self):
    if self.active or self.ball_in_flight:
      return False
    self.ball.cfg.active = True
    self.ball._resample_command(torch.tensor([0]))
    self.active = True
    self.ball_in_flight = self.ball_visible = True
    self.trial_fallen = False
    self.ball.cfg.freeze_on_done = self.scenario != "continuous"
    self.episode_start = self.elapsed
    self.observe_ball(update=True)
    return True

  @torch.inference_mode()
  def observe_ball(self, update=False):
    if update:
      self.ball._update_command()
    self.future = self.ball.future
    self.ball_obs = self.ball.command.clone()

  @torch.inference_mode()
  def set_mode(self, mode):
    controller = Controller(self.artifacts, self.motion, self.world.default_q, mode)
    self.controller, self.mode = controller, mode
    self.reset()

  @torch.inference_mode()
  def set_motion(self, path):
    if self.mode != "tracker":
      raise ValueError("Switch to Tracker mode before selecting a motion")
    path = Path(path).resolve()
    if path not in self.motion_catalog.values():
      raise ValueError("Motion is not in the available motion library")
    # Load and validate first: an invalid clip leaves the running controller intact.
    try:
      reference = Reference(path)
    except (OSError, ValueError, KeyError, EOFError, UnpicklingError) as error:
      raise ValueError(f"Cannot load motion {path.name}: {error}") from error
    self.controller.reference, self.motion = reference, path
    self.reset()
    self.paused = True

  @torch.inference_mode()
  def seek_motion(self, seconds):
    if self.mode != "tracker":
      raise ValueError("Only Tracker supports seeking the reference motion")
    self.reset(float(seconds))
    self.paused = True

  def adjacent_motion(self, direction):
    paths = [p for p in self.motion_catalog.values() if p.parent == self.motion.parent]
    return paths[(paths.index(self.motion) + direction) % len(paths)]

  def latch_control_fault(self, message):
    self.control_fault = message
    self.control_fault_events.append(
      {
        "time_s": self.elapsed,
        "step": self.records.total,
        "message": message,
        "response": "hold_last_valid_joint_targets_without_reset",
      }
    )
    qpos = torch.tensor(self.last_reference_qpos, dtype=torch.float32)
    root, quat = qpos[:3][None], qpos[3:7][None]
    state = torch.zeros(1, 72)
    state[:, 12:41] = qpos[self.world.qids]
    fk = self.controller.current.fk
    positions, quaternions = fk(root, quat, state[:, 12:41])
    self.controller.frozen_reference = SimpleNamespace(
      state=state, root=root, fk=fk, positions=positions, quaternions=quaternions
    )
    self.controller.info = {"policy": "FAULT: hold last valid targets"}

  @torch.inference_mode()
  def step(self):
    if self.max_record_steps and self.records.total >= self.max_record_steps:
      self.record_limit_reached = True
      self.paused = True
      return self.info
    if self.mode == "tracker" and self.tracker_at_end:
      if self.tracker_loop:
        self.reset()
      else:
        self.paused = True
        return self.info
    if self.needs_episode_reset:
      self.reset()
    if (
      not self.active
      and not self.ball_in_flight
      and not self.world.fallen
      and self.mode in ("rally", "hierarchy", "ball")
      and self.elapsed >= self.next_feed
    ):
      self.feed()
    self.observe_ball()
    if not self.control_fault:
      try:
        target = self.controller.step(
          self.world.robot_state, self.ball_obs, self.speed, self.latent
        )
        reference = self.controller.current
        if not all(
          torch.isfinite(value).all()
          for value in (
            reference.state,
            reference.positions,
            reference.quaternions,
          )
        ):
          raise RuntimeError("Non-finite reference pose")
        self.last_target = target.clone()
      except (RuntimeError, ValueError) as error:
        numerical_fault = any(
          word in str(error).lower()
          for word in (
            "non-finite",
            "nonfinite",
            "invalid latent/speed",
            "invalid tracker joint targets",
          )
        )
        if self.scenario != "continuous" or not numerical_fault:
          raise
        self.latch_control_fault(str(error))
    target = self.last_target
    kinematic = self.mode in ("mvae", "ball")
    if kinematic:
      self.world.reset(self.controller.current)
    for _ in range(4):
      if not kinematic:
        self.world.step(target[0].numpy())
      self.ball.compute_substep(self.world.dt)
      if self.scenario == "continuous" and self.ball_in_flight:
        at_floor = bool(self.ball.p[0, 2] <= self.ball.cfg.physics.radius)
        settled = bool(self.ball.done[0]) and float(self.ball.v.norm()) < 0.1
        if at_floor or settled:
          if at_floor:
            self.ball.p[:, 2].clamp_(min=self.ball.cfg.physics.radius)
          self.ball.v.zero_()
          self.ball.done[:] = True
          self.ball_in_flight = self.ball.cfg.active = False
          self.next_feed = self.elapsed + 0.5
    self.world.refresh()
    self.elapsed += 0.02
    self.observe_ball(update=True)
    if self.mode == "tracker":
      self.controller.reference_time = min(
        self.controller.reference_time + self.speed * 0.02,
        self.controller.reference.duration,
      )
      self.controller.reference.at(self.controller.reference_time)
      self.controller.tracker_delta = 0.02 * (self.speed - 1)
      self.tracker_at_end = (
        self.controller.reference_time >= self.controller.reference.duration - 1e-7
      )
      if self.tracker_at_end and not self.tracker_loop:
        self.paused = True
    self.ball.hit_event.zero_()
    self.ball.net_event.zero_()
    fallen = self.world.fallen
    self.trial_fallen |= self.active and fallen
    if fallen and not self.was_fallen:
      self.fall_events.append(
        {"time_s": self.elapsed, "step": self.records.total, "segment": self.segment}
      )
    self.was_fallen = fallen
    self.info = self.measure(target)
    if not self.control_fault:
      self.last_reference_qpos = self.info["reference_qpos"]
    self.records.append(self.info)
    ended = self.active and (
      bool(self.ball.done[0]) or (self.scenario == "independent" and fallen)
    )
    if ended:
      trial = {
        key: bool(getattr(self.ball, key)[0])
        for key in ("hit", "net_clear", "success", "double_hit", "illegal_contact")
      }
      trial = {
        **trial,
        "fallen": self.trial_fallen,
        "duration_s": self.elapsed - self.episode_start,
        "mode": self.mode,
      }
      self.trials.append(trial)
      self.active = False
      if self.scenario == "independent":
        self.ball_in_flight = self.ball.cfg.active = False
        self.needs_episode_reset = True
    if self.records.just_flushed:
      self.save(self.records.directory, completed=False)
    return self.info

  def measure(self, target):
    reference = self.controller.current
    wrist = reference.fk.names.index("right_wrist_yaw_link")
    center = reference.positions[:, wrist] + quat_apply(
      reference.quaternions[:, wrist], torch.tensor([G1_PINGPONG_MOUNT_OFFSET])
    )
    actual, normal = self.ball.paddle()
    result = {
      "time": self.elapsed,
      "motion": str(self.motion),
      "motion_time_s": self.controller.reference_time,
      "motion_duration_s": self.controller.reference.duration,
      "segment": self.segment,
      "mode": self.mode,
      "policy": self.controller.info.get("policy", self.mode),
      "control_fault": self.control_fault,
      "policy_output_valid": not bool(self.control_fault),
      "kinematic": self.mode in ("mvae", "ball"),
      "qpos": self.world.data.qpos.copy().tolist(),
      "qvel": self.world.data.qvel.copy().tolist(),
      "targets": target[0].tolist(),
      "reference_joints": reference.state[0, 12:41].tolist(),
      "ball": self.ball.p[0].tolist(),
      "ball_active": self.ball_visible,
      "trial_active": self.active,
      "ball_in_flight": self.ball_in_flight,
      "racket": actual[0].tolist(),
      "reference_racket": center[0].tolist(),
      "normal": normal[0].tolist(),
      "reference_qpos": reference_qpos(self.world, reference).tolist(),
      "alpha": float(self.controller.alpha[0]),
      "latent": self.controller.latent[0].tolist(),
      "joint_rms_rad": float(
        (self.world.robot_state.joint_positions - reference.state[:, 12:41])
        .square()
        .mean()
        .sqrt()
      ),
      "racket_error_m": float((actual - center).norm()),
      "fallen": self.world.fallen,
      "hit": bool(self.ball.hit[0]),
      "net_clear": bool(self.ball.net_clear[0]),
      "success": bool(self.ball.success[0]),
      "tracker_obs": self.controller.info.get(
        "tracker_observation", torch.zeros(1, 132)
      )[0].tolist(),
      "planner_obs": self.controller.info.get(
        "planner_observation", torch.zeros(1, 309)
      )[0].tolist(),
      "tracker_obs_valid": "tracker_observation" in self.controller.info,
      "planner_obs_valid": "planner_observation" in self.controller.info,
    }
    if (
      not np.isfinite(result["qpos"]).all() or not np.isfinite(result["targets"]).all()
    ):
      raise RuntimeError("Non-finite simulation/inference state")
    return result

  def save(self, directory, *, completed=True):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
      "schema": "g1-adapt-deploy-test-v2",
      "mode": self.mode,
      "scenario": self.scenario,
      "seed": self.seed,
      "steps": self.records.total,
      "simulated_seconds": self.records.total * self.world.control_dt,
      "trace_chunks": self.records.parts,
      "live_buffer_steps": len(self.records),
      "fall_events": self.fall_events,
      "control_fault_events": self.control_fault_events,
      "control_fault_active": bool(self.control_fault),
      "automatic_robot_reset": self.scenario == "independent",
      "automatic_pause_on_fall": False,
      "control_dt": 0.02,
      "physics_dt": self.world.dt,
      "backend": "native_cpu_mujoco",
      "motion": str(self.motion),
      "motion_events": self.motion_events,
      "motion_library": {k: str(p) for k, p in self.motion_catalog.items()},
      "random_initial_motion": self.random_initial_motion,
      "tracker_loop": self.tracker_loop,
      "asset_sha256": file_hash(G1_PINGPONG_XML),
      "physics": asdict(self.ball.cfg.physics),
      "noise": self.noise,
      "dropout": self.dropout,
      "delay_steps": self.delay_steps,
      "artifacts": self.artifacts.manifest(),
      "modules_used": self.controller.roles,
      "trials": self.trials,
      "completed": completed and not bool(getattr(self, "error", None)),
      "error": getattr(self, "error", None),
      "record_limit_reached": self.record_limit_reached,
      "max_record_steps": self.max_record_steps,
      "scope": "CPU deployment bridge; not MuJoCo-Warp training evaluation or hardware execution",
    }
    if self.records:
      summary["ever_fallen"] = self.records.ever_fallen
      summary["means"] = {
        key: value / self.records.total for key, value in self.records.sums.items()
      }
      self.records.write(output)
    if self.trials:
      summary["trial_rates"] = {
        key: float(np.mean([r[key] for r in self.trials]))
        for key in ("hit", "net_clear", "success", "fallen")
      }
    write_summary(output / "summary.json", summary)
    return summary

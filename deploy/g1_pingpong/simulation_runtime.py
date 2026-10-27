"""MuJoCo IO for run_real: identical control code, no SDK/ROS motor transport.

The clock/plant/sensors/operator are simulated; policy and state transitions
belong exclusively to run_real. No lifecycle state is assigned here.
"""

import copy
import json
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from deploy.backends.mujoco import MujocoBackend
from deploy.backends.playback import HELP as PLAYBACK_HELP
from deploy.backends.playback import PlaybackControl
from deploy.backends.runtime import SimulationIO, SimulationReset
from deploy.backends.unitree import run_real
from deploy.g1_pingpong.commands import policy_pd_settings
from deploy.g1_pingpong.core import quaternion_multiply, rotation
from deploy.g1_pingpong.feeder import RandomBallFeeder
from deploy.g1_pingpong.perception import BallEstimator, RootPoseCache
from deploy.io.diagnostics import (
  COMMAND_WRITE_DTYPE,
  LOWSTATE_DTYPE,
  BufferedWriter,
  CommandWriteRecorder,
  LowStateRecorder,
)
from deploy.io.transport import UDPMocap, build_mocap


class SimulatedMocap(UDPMocap):
  """Use the real normalized-packet decoder/estimators without opening a socket."""

  def __init__(self, config, clock, external_ball=False):
    self.cfg, self.clock = config, clock
    self.accepted = self.rejected = 0
    self.last_error = None
    self._planner_lock = threading.Lock()
    self._planner_packet = None
    self._planner_received = None
    self.root = RootPoseCache(
      config["root_marker_to_pelvis"],
      config["root_stale_timeout"],
      config.get("root_marker_to_pelvis_quaternion_wxyz", (1, 0, 0, 0)),
      conditioning=config.get("root_pose_filter"),
    )
    self.ball = BallEstimator(**config["estimator"])
    self.recorder = (
      BufferedWriter(Path(config["diagnostic_output_dir"]) / "mocap_packets.jsonl")
      if config.get("diagnostic_output_dir")
      else None
    )
    self.external = build_mocap(config) if external_ball else None
    if self.external:
      self.ball = SimpleNamespace(
        get=self.external_estimate,
        drain_observations=self.external.ball.drain_observations,
      )

  def external_estimate(self, now):
    sample = self.external.ball.get(time.monotonic())
    if sample is not None:
      sample.received = now - max(0, time.monotonic() - sample.received)
    return sample

  def publish(self, kind, position, quaternion=None):
    packet = dict(
      kind=kind,
      frame=self.cfg["world_frame"],
      stamp=self.clock.time() - self.cfg.get("source_clock_offset", 0),
      position=np.asarray(position).tolist(),
    )
    if quaternion is not None:
      packet["quaternion_wxyz"] = np.asarray(quaternion).tolist()
    self.accept(
      packet, wall_time=self.clock.time(), monotonic_time=self.clock.monotonic()
    )

  def diagnostics(self):
    now = self.clock.monotonic()
    return dict(
      accepted=self.accepted,
      rejected=self.rejected,
      last_error=self.last_error,
      root_valid=self.root.get(now) is not None,
      ball_valid=self.ball.get(now) is not None,
      transport="simulated normalized packets",
    )

  def close(self):
    if self.external:
      self.external.close()
    if self.recorder:
      self.recorder.close()


class MujocoRuntime:
  def __init__(
    self,
    deployment,
    runtime,
    options,
    *,
    controls,
    render,
    seed,
    episodes,
    standing_only,
    stable_seconds,
    policy_wait,
    final_hold,
    startup,
    ball_source,
    feeder_factory=RandomBallFeeder,
    _robot=None,
    _viewer=None,
  ):
    self._initial_options = dict(
      controls=controls,
      render=render,
      seed=seed,
      episodes=episodes,
      standing_only=standing_only,
      stable_seconds=stable_seconds,
      policy_wait=policy_wait,
      final_hold=final_hold,
      startup=startup,
      ball_source=ball_source,
      feeder_factory=feeder_factory,
    )
    self.deployment, self.runtime = deployment, runtime
    self.cfg = policy_pd_settings(deployment.settings, runtime)
    # The physical MuJoCo plant stays in its floor world, while simulated
    # mocap and the controller use the same HOPE world as live ROS inputs.
    self.world_translation = np.asarray(
      self.cfg.get("simulation_world_translation", [0.0, 0.0, 0.0])
    )
    if "simulation_ball" in self.cfg:
      self.cfg = {**self.cfg, "ball": self.cfg["simulation_ball"]}
    self.options = options
    self.controls, self.render, self.episodes = controls, render, episodes
    self.standing_only, self.stable_seconds, self.policy_wait, self.final_hold = (
      standing_only,
      stable_seconds,
      policy_wait,
      final_hold,
    )
    self.feed_interval = float(options["feed"].get("interval_seconds", 2.0))
    if not np.isfinite(self.feed_interval) or self.feed_interval <= 0:
      raise ValueError("Simulated feed interval must be finite and positive")
    self.continuous_feed = controls == "auto" or bool(
      options["feed"].get("continuous_on_policy_entry", False)
    )
    self.feed_references = None
    self.fallback_feed_references = []
    self.first_feed_time = None
    self.balls = []
    self.counted_hits = set()
    self.ball_source = ball_source
    self.robot = (
      _robot
      if _robot is not None
      else MujocoBackend(copy.copy(deployment.model), self.cfg)
    )
    if _robot is not None:
      self.robot.reset()
      self.robot.set_gains(self.cfg["robot"]["kp"], self.cfg["robot"]["kd"])
    self.engine = self.mocap = None
    self.viewer = _viewer
    self._viewer_threads = getattr(self, "_viewer_threads", ()) if _viewer else ()
    self.reset_requested = self.quit_requested = self.paused = self.step_requested = (
      False
    )
    self.ticks = self.sequence = self.command_sequence = self.write_sequence = 0
    self.pending = self.applied = self.last_submit_diagnostics = None
    self.pending_context = {}
    self.halted = self.closed = False
    self.lowstate_recorder = self.command_recorder = None
    self.next_write = self.next_sensor = 0.0
    self.previous_write = None
    self.last_written_id = 0
    self.last_recorded_state = None
    self.state_since = 0.0
    self.events = []
    self.physics_validation = dict(
      max_tilt_degrees=0.0, min_root_height_m=None, first_loss_of_standing_time=None
    )
    self.keys = []
    self.ball = None
    self.served_balls = []
    self.hits = self.served = 0
    self.next_serve = self.armed_since = self.finished_since = None
    self.force_ball = False
    self.automatic_up_issued = False
    self.support_active = startup == "suspended"
    self.support_z = 0.90
    self.landing_started = None
    self.wall_start = time.monotonic()
    self.playback = PlaybackControl(options.get("playback_rate", 1.))
    self.feeder = feeder_factory(
      self.cfg["ball"], options["feed"], np.random.default_rng(seed)
    )
    self.startup = startup
    self.metadata = dict(
      shared_control_loop="deploy.backends.unitree.run_real",
      clock="virtual simulation time",
      controls=controls,
      seed=seed,
      startup=startup,
      physics_dt=self.cfg["physics_dt"],
      policy_dt=self.cfg["control_dt"],
      writer_period=0.002,
      mocap_period=0.01,
      feed_interval_seconds=self.feed_interval,
      feed_schedule="independent periodic arrivals; no stance readiness gate",
      actual_robot_commands_sent=False,
      physics_world_frame="mujoco_floor",
      control_world_frame=self.cfg.get("world_frame", "world"),
      mujoco_to_control_translation=self.world_translation.tolist(),
      actual_network_connections=ball_source == "mocap",
      ideal_io=True,
      physical_limit_mode="observe_only; policy and physics continue",
      notes="No hardware latency/drive identification is implied. Virtual CPU/IPC timings are not host measurements.",
      events=self.events,
    )
    if render and self.viewer is None:
      from mujoco import viewer

      previous_threads = set(threading.enumerate())
      camera = options.get("viewer", {})
      self.viewer = viewer.launch_passive(
        self.robot.model,
        self.robot.data,
        key_callback=self.key,
        show_left_ui=camera.get("show_left_ui", True),
        show_right_ui=camera.get("show_right_ui", True),
      )
      with self.viewer.lock():
        if "lookat" in camera:
          self.viewer.cam.lookat[:] = camera["lookat"]
        for name in ("distance", "elevation", "azimuth"):
          if name in camera:
            setattr(self.viewer.cam, name, camera[name])
      # launch_passive returns before its daemon renderer exits. Retain the
      # threads created by this launch so Q can wait for GL cleanup before
      # Python's atexit GLFW termination runs.
      self._viewer_threads = tuple(
        thread for thread in threading.enumerate() if thread not in previous_threads
      )
      print(
        "MuJoCo keys: ENTER=START, SPACE/P=pause/resume, L=lower, A=AMP, W=UP/pingpong, S=DOWN/AMP, F=ball, T=continuous feed, R=reset, N=single step, B/X=halt, Q=quit, H=help",
        flush=True,
      )
      print(f"MuJoCo playback: {PLAYBACK_HELP}", flush=True)

  def monotonic(self):
    return self.ticks * self.cfg["physics_dt"]

  def time(self):
    return 10000 + self.monotonic()

  def key(self, key):
    if self.playback.key(key):
      return
    if ord("a") <= key <= ord("z"):
      key -= 32
    if key == ord("R"):
      self.reset_requested = True
      self.paused = False
      return
    if key in (ord(" "), ord("P")):
      self.paused = not self.paused
      self.step_requested = False
      print("[MuJoCo] paused" if self.paused else "[MuJoCo] resumed", flush=True)
      return
    if key == ord("N"):
      self.step_requested = True
      return
    if key == ord("H"):
      print(
        "ENTER=START, SPACE/P=pause/resume, L=lower, A=AMP, W=UP/pingpong, S=DOWN/AMP, F=ball, T=continuous feed, R=reset, N=single step, B/X=halt, Q=quit",
        flush=True,
      )
      print(f"Playback {self.playback.rate:g}x: {PLAYBACK_HELP}", flush=True)
      return
    if key in (ord("Q"), ord("B"), ord("X")):
      self.paused = False
      if key == ord("Q"):
        self.quit_requested = True
    if key in (ord("L"), ord("l")):
      self.begin_landing()
      return
    if key == ord("T"):
      self.continuous_feed = not self.continuous_feed
      if not self.continuous_feed:
        self.next_serve = None
      self.next_serve = None
      print(
        f"[MuJoCo] continuous feed: {self.continuous_feed}, interval={self.feed_interval}s",
        flush=True,
      )
      return
    if key in (ord("F"), ord("f")):
      if self.engine is not None and self.engine.lifecycle.state in (
        "ARMED",
        "EXECUTE",
        "RECOVER",
        "READY",
      ):
        self.force_ball = True
      else:
        print("[MuJoCo] F ignored: enter pingpong policy first", flush=True)
      return
    mapping = {
      257: 1 << 2,  # GLFW_KEY_ENTER
      335: 1 << 2,  # GLFW_KEY_KP_ENTER
      ord("A"): 1 << 8,
      ord("W"): 1 << 12,
      ord("S"): 1 << 14,
      ord("Q"): 1 << 3,
      ord("B"): 1 << 9,
      ord("X"): 1 << 10,
    }
    if key in mapping:
      self.keys.append(mapping[key])
      if key == ord("S"):
        self.automatic_up_issued = False

  def reset(self):
    """Fresh plant and IO on the same viewer; run_real rebuilds every policy."""
    rate = self.playback.rate
    with self.viewer.lock() if self.viewer else nullcontext():
      self.__init__(
        self.deployment,
        self.runtime,
        self.options,
        **self._initial_options,
        _robot=self.robot,
        _viewer=self.viewer,
      )
      self.playback.set_rate(rate)

  def close_viewer(self):
    if self.viewer:
      self.viewer.close()
      for thread in self._viewer_threads:
        if thread is not threading.current_thread():
          thread.join(timeout=5)
          if thread.is_alive():
            raise TimeoutError("MuJoCo renderer did not finish shutdown")

  def check_ui(self):
    before = time.monotonic()
    while self.paused and not self.step_requested and not self.reset_requested:
      if self.quit_requested or (self.viewer and not self.viewer.is_running()):
        break
      if self.viewer:
        self.viewer.sync()
      time.sleep(0.01)
    self.wall_start += time.monotonic() - before
    if self.paused or time.monotonic() - before > .005:
      self.playback.reset(self.monotonic())
    self.step_requested = False
    if self.reset_requested:
      raise SimulationReset()

  def bind(self, engine):
    self.engine = engine

  def open_mocap(self, config):
    self.mocap = SimulatedMocap(config, self, self.ball_source == "mocap")
    self.publish_root()
    return self.mocap

  def open_robot(self, cfg, runtime, enabled):
    assert enabled
    folder = runtime.get("diagnostic_output_dir")
    if folder:
      names = self.cfg["robot"]["joint_names"]
      if runtime.get("record_lowstate_hz", 0) > 0:
        self.lowstate_recorder = LowStateRecorder(
          Path(folder) / "lowstate", names, runtime["record_lowstate_hz"]
        )
      if runtime.get("record_command_writes", False):
        self.command_recorder = CommandWriteRecorder(
          Path(folder) / "command_writes", names
        )
      for recorder in [self.lowstate_recorder, self.command_recorder]:
        if recorder is not None:
          p = recorder.path / "schema.json"
          schema = json.loads(p.read_text())
          schema.update(
            clock="virtual simulation seconds",
            transport="MuJoCo only; NOT DDS",
            sampling="simulated sensor samples or simulated 500Hz target application",
            actual_robot_commands_sent=False,
          )
          p.write_text(json.dumps(schema, indent=2))
    return self

  def buttons(self):
    if self.viewer and not self.viewer.is_running():
      return 1 << 3
    if self.keys:
      return self.keys.pop(0)
    if self.controls != "auto":
      return 0
    now = self.monotonic()
    phase = self.engine.lifecycle.state
    if phase == "WAIT" and now - self.state_since >= 0.5:
      return 1 << 2
    if phase == "POSE_HOLD":
      if self.support_active:
        self.begin_landing()
        return 0
      # Suspended startup already holds PD throughout the 1.5 s lowering.
      # Request AMP on the first feedback cycle after fixture release; waiting
      # for a separate pose-hold timer leaves the grounded robot on static PD.
      if self.landing_started is not None or now - self.state_since >= 2:
        return 1 << 8
    if (
      phase == "STABLE"
      and not self.automatic_up_issued
      and now - self.state_since >= self.stable_seconds
    ):
      self.automatic_up_issued = True
      return 1 << 12
    if self.finished_since is not None and now - self.finished_since >= self.final_hold:
      return 1 << 3
    return 0

  def feedback(self):
    state = self.robot.state()
    return (
      state.q,
      state.dq,
      state.gyro,
      self.buttons(),
      self.runtime["expected_mode_machine"],
      state.root_quat,
      self.monotonic(),
    )

  def diagnostics(self):
    out = dict(
      feedback_sequence=self.ticks,
      state_source_tick=round(self.monotonic() * 1000),
      state_callback_time=self.monotonic(),
      transport="MuJoCo; no DDS",
      virtual_time=True,
      simulation_fixture_active=self.support_active,
      landing_remaining_seconds=(
        max(0.0, 1.5 - (self.monotonic() - self.landing_started))
        if self.landing_started is not None
        else None
      ),
    )
    if self.lowstate_recorder:
      out["lowstate_recording"] = self.lowstate_recorder.diagnostics()
    if self.command_recorder:
      out["command_write_recording"] = self.command_recorder.diagnostics()
    return out

  def note_command_context(self, context):
    self.pending_context = dict(context)

  def send(self, target, kp=None, kd=None):
    self.command_sequence += 1
    now = self.monotonic()
    self.pending = dict(
      target=np.asarray(target).copy(),
      kp=np.asarray(kp).copy(),
      kd=np.asarray(kd).copy(),
      context=self.pending_context.copy(),
      generated=now,
      id=self.command_sequence,
    )
    self.pending_context = {}
    self.last_submit_diagnostics = dict(
      command_id=self.command_sequence,
      command_generated_time=now,
      enqueue_completed_time=now,
      command_queue_put_succeeded=True,
    )
    return now

  def apply_command(self):
    if self.pending is None:
      return
    c = self.pending
    now = self.monotonic()
    first = self.last_written_id != c["id"]
    self.last_written_id = c["id"]
    self.write_sequence += 1
    self.robot.set_gains(c["kp"], c["kd"])
    self.robot.set_target(c["target"])
    self.applied = c
    if self.command_recorder:
      r = np.zeros((), dtype=COMMAND_WRITE_DTYPE)
      ctx = c["context"]
      for name in COMMAND_WRITE_DTYPE.names:
        if (
          COMMAND_WRITE_DTYPE[name].kind == "f"
          and COMMAND_WRITE_DTYPE[name].shape == ()
        ):
          r[name] = np.nan
      for key in [
        "feedback_callback_time",
        "feedback_consumed_time",
        "feedback_packet_enqueued_time",
        "feedback_packet_dequeued_time",
        "feedback_sequence",
        "control_cycle",
        "source_tick",
      ]:
        if key in ctx:
          r[key] = ctx[key]
      for key in [
        "accepted_time",
        "prepare_started_time",
        "scheduled_time",
        "write_started_time",
        "write_completed_time",
        "writer_feedback_callback_time",
        "command_dequeued_time",
      ]:
        r[key] = now
      r["generated_time"] = c["generated"]
      r["write_interval_seconds"] = (
        now - self.previous_write if self.previous_write is not None else np.nan
      )
      r["write_sequence"] = self.write_sequence
      r["command_id"] = c["id"]
      r["writer_source_tick"] = round(now * 1000)
      r["first_write"] = first
      r["write_result"] = -1
      r["q_target"] = c["target"]
      r["kp"] = c["kp"]
      r["kd"] = c["kd"]
      self.command_recorder.write(r.copy())
    self.previous_write = now

  def publish_root(self):
    state = self.robot.state()
    cfg = self.mocap.cfg
    # The decoder applies marker->pelvis. Generate the corresponding marker
    # sample, so nonzero calibration is neither ignored nor applied twice.
    offset_q = np.asarray(
      cfg.get("root_marker_to_pelvis_quaternion_wxyz", [1, 0, 0, 0])
    )
    inverse = offset_q * np.array([1, -1, -1, -1])
    marker_q = quaternion_multiply(state.root_quat, inverse)
    marker_p = state.root_pos - rotation(marker_q) @ np.asarray(
      cfg["root_marker_to_pelvis"]
    )
    self.mocap.publish(
      "root", marker_p + getattr(self, "world_translation", np.zeros(3)), marker_q
    )

  def begin_landing(self):
    if self.support_active and self.landing_started is None:
      self.landing_started = self.monotonic()
      self.events.append(dict(time=self.monotonic(), event="operator_lowering_fixture"))
      print("[MuJoCo] lowering support (1.5s)", flush=True)

  def support(self):
    if not self.support_active:
      return
    r = self.robot
    i, v = r.free_qpos, r.free_qvel
    if self.landing_started is not None:
      # Same placement height as the validated G1 standing fixture. This is
      # explicitly an operator fixture, never a policy-state correction.
      alpha = min(1, (self.monotonic() - self.landing_started) / 1.5)
      self.support_z = 0.90 + alpha * (0.783675 - 0.90)
      if alpha >= 1:
        self.support_active = False
        self.events.append(
          dict(time=self.monotonic(), event="operator_fixture_released")
        )
        print("[MuJoCo] support released", flush=True)
        return
    r.data.qpos[i : i + 3] = [0, 0, self.support_z]
    r.data.qpos[i + 3 : i + 7] = [1, 0, 0, 0]
    r.data.qvel[v : v + 6] = 0
    mujoco.mj_forward(r.model, r.data)

  def sample_lowstate(self):
    if not self.lowstate_recorder or not self.lowstate_recorder.due(self.monotonic()):
      return
    state = self.robot.state()
    r = np.zeros((), dtype=LOWSTATE_DTYPE)
    r["callback_time"] = self.monotonic()
    r["source_tick"] = round(self.monotonic() * 1000)
    r["mode_machine"] = self.runtime["expected_mode_machine"]
    r["q"] = state.q
    r["dq"] = state.dq
    r["gyro"] = state.gyro
    r["imu_quat_wxyz"] = state.root_quat
    r["tau_est"] = self.robot.data.actuator_force[self.robot.actuators]
    r["voltage"] = np.nan
    r["callback_sequence"] = self.ticks
    r["last_write_sequence"] = self.write_sequence
    r["last_written_command_id"] = self.last_written_id
    if self.applied:
      r["last_written_origin"] = self.applied["generated"]
    r["last_write_started_time"] = r["last_write_completed_time"] = (
      self.previous_write if self.previous_write is not None else np.nan
    )
    self.lowstate_recorder.write(r.copy())

  def sleep(self, seconds):
    self.check_ui()
    count = round(seconds / self.cfg["physics_dt"])
    for _ in range(max(0, count)):
      if self.monotonic() + 1e-9 >= self.next_write:
        self.apply_command()
        self.next_write += 0.002
      self.robot.substep()
      self.ticks += 1
      self.support()
      state = self.robot.state() if self.balls else None
      for ball in self.balls:
        ball.substep(state, self.cfg["physics_dt"])
        if ball.hit and ball.feed_id not in self.counted_hits:
          self.counted_hits.add(ball.feed_id)
          self.hits += 1
      self.balls = [
        ball
        for ball in self.balls
        if ball is self.ball or ball.elapsed < ball.arrival + 3
      ]
      if self.monotonic() + 1e-9 >= self.next_sensor:
        self.publish_root()
        if self.ball:
          self.mocap.publish("ball", self.ball.position + self.world_translation)
        self.next_sensor += 0.01
      self.sample_lowstate()
    if self.render or self.ball_source == "mocap":
      self.playback.wait(self.monotonic())

  def record(self, row):
    now = self.monotonic()
    state = row["state"]
    if row.get("command_policy"):
      measured = self.robot.state()
      tilt = float(
        np.degrees(np.arccos(np.clip(rotation(measured.root_quat)[2, 2], -1, 1)))
      )
      height = float(measured.root_pos[2])
      stats = self.physics_validation
      stats["max_tilt_degrees"] = max(stats["max_tilt_degrees"], tilt)
      stats["min_root_height_m"] = (
        min(stats["min_root_height_m"], height)
        if stats["min_root_height_m"] is not None
        else height
      )
      if stats["first_loss_of_standing_time"] is None and (tilt > 45 or height < 0.4):
        stats["first_loss_of_standing_time"] = now
        self.events.append(
          dict(
            time=now,
            event="loss_of_standing_observed",
            state=state,
            tilt_degrees=tilt,
            root_height_m=height,
          )
        )
    if state != self.last_recorded_state:
      self.state_since = now
    self.last_recorded_state = state
    if (
      self.engine.recovered >= self.episodes
      and not self.standing_only
      and self.finished_since is None
    ):
      self.finished_since = now
    active_policy = (
      state in ("ARMED", "EXECUTE", "RECOVER", "READY") and self.engine.receiving
    )
    can_feed = (
      active_policy
      and not self.standing_only
      and self.ball_source == "sim"
      and (self.controls == "manual" or self.engine.completed < self.episodes)
    )
    if not active_policy:
      self.next_serve = self.first_feed_time = None
      self.feed_references = None
    if can_feed:
      if self.feed_references is None:
        # The opponent aims at the court's initial receive region, independently
        # of subsequent robot sway, falls, recovery or readiness metrics.
        self.feed_references = []
        self.fallback_feed_references = []
        if getattr(self.feeder, "independent_launch", False):
          self.events.append(
            dict(
              time=now, event="independent_launch_region", region=self.options["feed"]
            )
          )
        else:
          ball_cfg = self.cfg["ball"]
          control_state = copy.copy(self.robot.state())
          translation = getattr(self, "world_translation", np.zeros(3))
          if np.any(translation):
            for name in ("root_pos", "anchor_pos", "racket_pos"):
              setattr(control_state, name, getattr(control_state, name) + translation)
          for index, reference in self.engine.candidates(control_state):
            if np.any(translation):
              # Convert reference geometry back for the physical feeder. Never
              # change the controller's frozen home or motion transforms.
              control_reference = reference
              offset = translation.copy()
              reference = SimpleNamespace(
                source=getattr(control_reference, "source", control_reference),
                rotation=getattr(control_reference, "rotation", np.eye(3)),
                fps=control_reference.fps,
                strike_position=lambda cfg, ref=control_reference, offset=offset: (
                  ref.strike_position(cfg) - offset
                ),
                strike_time=control_reference.strike_time,
              )
            point = reference.strike_position(ball_cfg).copy()
            # A below-table strike point rotated over the tabletop cannot be
            # reached by any legal incoming flight. Serve just beyond the near
            # edge instead, without waiting for the robot to turn/stand upright.
            if point[2] <= ball_cfg["table_height"] + ball_cfg["ball_radius"]:
              point[0] = min(
                point[0], ball_cfg["table_near_x"] - ball_cfg["ball_radius"] - 0.03
              )
            point[1] = np.clip(
              point[1],
              -ball_cfg["table_width"] / 2 + 0.1,
              ball_cfg["table_width"] / 2 - 0.1,
            )
            fixed = SimpleNamespace(
              strike_position=lambda cfg, point=point: point.copy(),
              strike_time=reference.strike_time,
            )
            self.feed_references.append((index, reference))
            self.fallback_feed_references.append((index, fixed))
          self.events.append(
            dict(
              time=now,
              event="fixed_court_feed_region",
              points=[
                r.strike_position(ball_cfg).tolist() for _, r in self.feed_references
              ],
            )
          )
        self.first_feed_time = now + self.policy_wait
      if self.continuous_feed and self.next_serve is None:
        self.next_serve = max(now, self.first_feed_time)
      if self.force_ball:
        self.next_serve = now
        self.force_ball = False
      if self.next_serve is not None and now >= self.next_serve:
        try:
          try:
            incoming = self.feeder.sample(self.feed_references)
          except ValueError:
            if not self.fallback_feed_references or getattr(
              self.feeder, "requires_motion_geometry", False
            ):
              raise
            incoming = self.feeder.sample(self.fallback_feed_references)
            self.feed_references = self.fallback_feed_references
            self.events.append(
              dict(
                time=now,
                event="court_feeder_fallback",
                reason="initial receive region had no legal incoming flight",
              )
            )
        except ValueError as exc:
          self.events.append(
            dict(time=now, event="no_legal_simulated_feed", error=str(exc))
          )
          self.next_serve = now + min(0.1, self.feed_interval)
        else:
          self.served += 1
          incoming.feed_id = self.served
          incoming.previous_racket = self.robot.state().racket_pos.copy()
          self.ball = incoming
          self.served_balls.append(incoming)
          self.balls.append(incoming)
          previous_estimate = self.mocap.ball.get(now)
          self.mocap.ball = BallEstimator(**self.mocap.cfg["estimator"])
          # A new incoming ball needs three position samples for velocity.
          # Keep the last valid measured estimate with its ORIGINAL timestamp
          # while that fit warms up; do not invent velocity or refresh its age.
          self.mocap.ball.latest = previous_estimate
          self.next_serve = now + self.feed_interval if self.continuous_feed else None
          self.events.append(
            dict(
              time=now,
              event="simulated_ball",
              index=self.served,
              control_state=state,
              independent_of_readiness=True,
            )
          )
    elif not active_policy:
      self.force_ball = False
    if self.viewer:
      with self.viewer.lock():
        self.viewer.user_scn.ngeom = 0
        for ball in self.balls:
          if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
            break
          geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
          mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, self.cfg["ball"]["ball_radius"]),
            ball.position,
            np.eye(3).ravel(),
            np.array([1.0, 0.6, 0.1, 1.0]),
          )
          self.viewer.user_scn.ngeom += 1
        if hasattr(self.feeder, "draw_grid"):
          self.feeder.draw_grid(self.viewer.user_scn)
      self.viewer.set_texts((None, None, "Playback\n[ / - slower\n] / + faster\n\\ normal",
                             f"{self.playback.rate:g}x"))
      self.viewer.sync()

  def close(self):
    self.closed = True
    self.robot.damp(self.runtime["halt_kd"])
    if self.lowstate_recorder:
      self.lowstate_recorder.close()
    if self.command_recorder:
      self.command_recorder.close()


def run_deployment(
  deployment,
  real_config,
  mocap_config,
  *,
  simulation_config="deploy/configs/mujoco.yaml",
  motion_index=19,
  motion_select="auto",
  motion_allow=None,
  duration=120,
  episodes=3,
  seed=42,
  render=False,
  controls="auto",
  standing_only=False,
  stable_seconds=3,
  policy_wait=0,
  final_hold_seconds=5,
  startup="suspended",
  physics_dt=0.001,
  ball_source="sim",
  feed_interval=None,
  output=None,
  controller_factory=None,
  minimum_hit_rate=None,
  minimum_return_rate=None,
  feeder_factory=RandomBallFeeder,
):
  if minimum_return_rate is not None and (
    not np.isfinite(minimum_return_rate)
    or not 0 <= minimum_return_rate <= 1
    or ball_source != "sim"
    or standing_only
    or controls != "auto"
  ):
    raise ValueError(
      "Minimum return rate requires automatic simulated-ball receiving and a value in [0, 1]"
    )
  if minimum_hit_rate is not None and (
    not np.isfinite(minimum_hit_rate)
    or not 0 <= minimum_hit_rate <= 1
    or ball_source != "sim"
    or standing_only
    or controls != "auto"
  ):
    raise ValueError(
      "Minimum hit rate requires automatic simulated-ball receiving and a value in [0, 1]"
    )
  if controls == "manual" and not render:
    raise ValueError("Manual MuJoCo controls require --render")
  if (
    not np.isfinite(physics_dt)
    or physics_dt <= 0
    or not np.isclose(0.002 / physics_dt, round(0.002 / physics_dt))
    or not np.isclose(
      deployment.settings["control_dt"] / physics_dt,
      round(deployment.settings["control_dt"] / physics_dt),
    )
  ):
    raise ValueError(
      "Simulation physics_dt must divide both the 2ms writer and policy period"
    )
  if min(stable_seconds, policy_wait, final_hold_seconds) < 0:
    raise ValueError("Simulation dwell durations cannot be negative")
  runtime = yaml.safe_load(Path(real_config).read_text())
  cfg = copy.deepcopy(deployment.settings)
  cfg["physics_dt"] = physics_dt
  cfg["decimation"] = round(cfg["control_dt"] / physics_dt)
  cfg["sim_options"]["timestep"] = physics_dt
  active = replace(deployment, settings=cfg)
  options = yaml.safe_load(Path(simulation_config).read_text())
  if feed_interval is not None:
    options["feed"]["interval_seconds"] = feed_interval
  session = MujocoRuntime(
    active,
    runtime,
    options,
    controls=controls,
    render=render,
    seed=seed,
    episodes=episodes,
    standing_only=standing_only,
    stable_seconds=stable_seconds,
    policy_wait=policy_wait,
    final_hold=final_hold_seconds,
    startup=startup,
    ball_source=ball_source,
    feeder_factory=feeder_factory,
  )
  if controller_factory is not None:
    session.metadata["shared_control_loop"] = "deploy.backends.unitree.run_control_loop"
    session.metadata["task"] = cfg["task"]
    session.metadata["controller"] = (
      controller_factory.__module__ + "." + controller_factory.__name__
    )
  attempts = []
  attempt = 0
  base_output = Path(output) if output else None
  try:
    while True:
      attempt_output = base_output
      if attempt and base_output:
        attempt_output = (
          base_output.parent / f"{base_output.stem}_reset_{attempt:03d}.jsonl"
          if base_output.suffix == ".jsonl"
          else base_output / f"reset_{attempt:03d}"
        )
      result = _run_attempt(
        active,
        runtime,
        mocap_config,
        session=session,
        simulation_config=simulation_config,
        motion_index=motion_index,
        motion_select=motion_select,
        motion_allow=motion_allow,
        duration=duration,
        output=attempt_output,
        controller_factory=controller_factory,
        minimum_hit_rate=minimum_hit_rate,
        minimum_return_rate=minimum_return_rate,
      )
      attempts.append(
        dict(
          attempt=attempt,
          output=str(attempt_output) if attempt_output else None,
          termination=result.get("termination"),
          passed=result["passed"],
          error=result["error"],
        )
      )
      if base_output:
        index_path = (
          base_output.with_suffix(".session.json")
          if base_output.suffix == ".jsonl"
          else base_output / "interactive_session.json"
        )
        index_path.write_text(
          json.dumps(
            dict(
              attempts=attempts,
              actual_robot_commands_sent=False,
              reset_semantics="full plant, IO and policy reinitialization; same random seed",
            ),
            indent=2,
          )
        )
      # Explicit stop or an unrecoverable software error ends the controller,
      # but never freezes physical motion. The already-damped plant continues.
      if render and controls == "manual" and not session.reset_requested:
        if result.get("termination") != "interrupted" and not session.quit_requested:
          print(
            "[MuJoCo] controller ended; damping physics continues; R=reset, Q=quit",
            flush=True,
          )
          post_writer = None
          if attempt_output:
            post_path = (
              attempt_output.with_suffix(".post_control.jsonl")
              if attempt_output.suffix == ".jsonl"
              else attempt_output / "post_control_physics.jsonl"
            )
            post_writer = BufferedWriter(post_path)
          try:
            while (
              session.viewer.is_running()
              and not session.reset_requested
              and not session.quit_requested
            ):
              before = time.monotonic()
              try:
                session.check_ui()
              except SimulationReset:
                break
              for _ in range(session.cfg["decimation"]):
                session.robot.substep()
                session.ticks += 1
                for ball in session.balls:
                  ball.substep(session.robot.state(), session.cfg["physics_dt"])
              if post_writer:
                state = session.robot.state()
                post_writer.write(
                  json.dumps(
                    dict(
                      time=session.monotonic(),
                      control_mode="damping",
                      policy_running=False,
                      q=state.q.tolist(),
                      dq=state.dq.tolist(),
                      root_pos=state.root_pos.tolist(),
                      root_quat=state.root_quat.tolist(),
                      actual_robot_commands_sent=False,
                    )
                  )
                  + "\n"
                )
              if session.balls:
                with session.viewer.lock():
                  for index, ball in enumerate(
                    session.balls[: session.viewer.user_scn.ngeom]
                  ):
                    session.viewer.user_scn.geoms[index].pos[:] = ball.position
              session.viewer.sync()
              session.playback.wait(session.monotonic(), frame_started=before)
          finally:
            if post_writer:
              post_writer.close()
      if (
        not session.reset_requested
        or session.quit_requested
        or (session.viewer and not session.viewer.is_running())
      ):
        break
      attempt += 1
      print(
        f"[MuJoCo] RESET {attempt}: new WAIT, reset plant, policy histories and sensors",
        flush=True,
      )
      session.reset()
    result["attempt_count"] = len(attempts)
    return result
  except KeyboardInterrupt:
    return dict(passed=False, error="interrupted", attempts=attempts)
  finally:
    session.close_viewer()


def _run_attempt(
  active,
  runtime,
  mocap_config,
  *,
  session,
  simulation_config,
  motion_index,
  motion_select,
  motion_allow,
  duration,
  output,
  controller_factory=None,
  minimum_hit_rate=None,
  minimum_return_rate=None,
):
  controls, standing_only, episodes, ball_source = (
    session.controls,
    session.standing_only,
    session.episodes,
    session.ball_source,
  )
  adapter = SimulationIO(
    session,
    session.open_robot,
    session.open_mocap,
    session.bind,
    session.record,
    session.metadata,
    observe_physical_limits=True,
  )
  error = None
  result = None
  try:
    loop = run_real
    controller_options = {}
    if controller_factory is not None:
      from deploy.backends.unitree import run_control_loop

      loop = run_control_loop
      controller_options["controller_factory"] = controller_factory
    result = loop(
      active,
      runtime,
      mocap_config,
      enable_motors=True,
      motion_index=motion_index,
      motion_select=motion_select,
      motion_allow=motion_allow,
      duration=duration,
      output=output,
      simulation_config=simulation_config,
      simulation_io=adapter,
      **controller_options,
    )
  except SimulationReset:
    pass
  except (ValueError, TimeoutError, RuntimeError) as exc:
    error = str(exc)
  summary_path = (
    (
      Path(output).with_suffix(".summary.json")
      if Path(output).suffix == ".jsonl"
      else Path(output) / "summary.json"
    )
    if output
    else None
  )
  result = result or (
    json.loads(summary_path.read_text())
    if summary_path and summary_path.exists()
    else dict(
      failure=None if error is None else dict(error=error),
      phases=[],
      completed_swings=session.engine.completed,
      recovered_swings=session.engine.recovered,
    )
  )
  reached = "ARMED" in result["phases"]
  flow_passed = (
    error is None
    and not session.reset_requested
    and reached
    and not result.get("physical_limit_observation_counts")
    and session.physics_validation["first_loss_of_standing_time"] is None
    and (standing_only or controls == "manual" or session.engine.recovered >= episodes)
  )
  hit_rate = session.hits / session.served if session.served else 0.0
  ball_returns = [ball.return_report() for ball in session.served_balls]
  legal_returns = sum(ball["legal_return"] for ball in ball_returns)
  return_rate = legal_returns / session.served if session.served else 0.0
  passed = (
    flow_passed
    and (minimum_hit_rate is None or hit_rate >= minimum_hit_rate)
    and (minimum_return_rate is None or return_rate >= minimum_return_rate)
  )
  result.update(
    passed=bool(passed),
    flow_passed=bool(flow_passed),
    hit_rate=hit_rate,
    minimum_hit_rate=minimum_hit_rate,
    minimum_return_rate=minimum_return_rate,
    robot_resets=1,
    reset_requested=session.reset_requested,
    actual_robot_commands_sent=False,
    actual_network_connections=ball_source == "mocap",
    simulation=session.metadata,
    physics_validation=session.physics_validation,
    simulated_hits=session.hits,
    simulated_balls=session.served,
    simulated_legal_returns=legal_returns,
    legal_return_rate=return_rate,
    ball_returns=ball_returns,
    error=error,
  )
  if summary_path:
    summary_path.write_text(json.dumps(result, indent=2))
  print(
    json.dumps(
      dict(
        passed=result["passed"],
        error=error,
        completed=result.get("completed_swings"),
        recovered=result.get("recovered_swings"),
        hits=session.hits,
        shared_control_loop=session.metadata["shared_control_loop"],
      )
    ),
    flush=True,
  )
  return result

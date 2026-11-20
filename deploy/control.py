"""Module-specific inference using the current AdaPT motion/observation code."""

from typing import Any

import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_TRACKED_BODIES,
)
from mjlab.tasks.g1_adapt.actions import decode_speed
from mjlab.tasks.g1_adapt.generator import MotionGenerator
from mjlab.tasks.g1_adapt.motion import G1Kinematics, pack_state
from mjlab.tasks.g1_adapt.mvae import load_mvae
from mjlab.tasks.g1_adapt.runtime import observation_prefix
from mjlab.tasks.g1_stage1.mdp.commands import MultiMotionLoader
from mjlab.tasks.g1_stage1.mdp.legacy_schema import (
  LEGACY_G1_BODY_NAMES,
  LEGACY_G1_JOINT_NAMES,
)

MODES = ("tracker", "mvae", "mvae_tracker", "rally", "standing", "hierarchy", "ball")


class Reference:
  def __init__(self, path):
    self.fk = G1Kinematics("cpu")
    self.library = MultiMotionLoader(
      (str(path),),
      torch.arange(len(G1_PINGPONG_TRACKED_BODIES)),
      "cpu",
      target_body_names=G1_PINGPONG_TRACKED_BODIES,
      target_joint_names=G1_PINGPONG_JOINT_NAMES,
      legacy_body_names=LEGACY_G1_BODY_NAMES,
      legacy_joint_names=LEGACY_G1_JOINT_NAMES,
      align_heading_to_frame="none",
    )
    self.duration = (int(self.library.lengths[0]) - 1) / float(
      self.library.fps_values[0]
    )
    self.ids = torch.zeros(1, dtype=torch.long)
    self.tracked_ids = [self.fk.names.index(n) for n in G1_PINGPONG_TRACKED_BODIES]
    with np.load(path, allow_pickle=True) as archive:
      self.strike = float(
        np.asarray(archive.get("strike_time_s", 0.46 * self.duration)).reshape(-1)[0]
      )
    self.at(0)

  def at(self, seconds):
    t = torch.tensor([min(max(seconds, 0), self.duration)], dtype=torch.float32)

    def get(key):
      return self.library._gather_interp(key, self.ids, t)

    p, q = get("body_pos_w"), get("body_quat_w")
    angle = (
      torch.pi * t / max(self.strike, 0.02)
      if seconds <= self.strike
      else torch.pi
      + torch.pi * (t - self.strike) / max(self.duration - self.strike, 0.02)
    )
    phase = torch.stack((angle.cos(), angle.sin()), -1)
    self.history = pack_state(
      get("joint_pos"),
      get("joint_vel"),
      q[:, 0],
      get("body_lin_vel_w")[:, 0],
      get("body_ang_vel_w")[:, 0],
      phase,
    )[:, None]
    self.root = p[:, 0]
    self.positions, self.quaternions = self.fk(self.root, q[:, 0], self.state[:, 12:41])
    self.positions[:, self.tracked_ids] = p
    self.quaternions[:, self.tracked_ids] = q
    return self

  @property
  def state(self):
    return self.history[:, -1]


class Controller:
  def __init__(self, artifacts, motion, default_q, mode):
    self.mode = mode
    self.reference = Reference(motion)
    self.default_q = default_q
    self.scales = torch.tensor(
      [G1_PINGPONG_ACTION_SCALE[n] for n in G1_PINGPONG_JOINT_NAMES]
    )
    roles = []
    if mode not in ("mvae", "ball"):
      roles.append("tracker")
    if mode not in ("tracker", "ball"):
      roles.append("mvae")
    if mode in ("rally", "standing"):
      roles.append(mode)
    if mode == "hierarchy":
      roles += ["rally", "standing"]
    artifacts.require(roles)
    self.roles = tuple(roles)
    self.tracker = (
      torch.jit.load(str(artifacts.files["tracker"])).eval()
      if "tracker" in roles
      else None
    )
    self.planners = {
      name: torch.jit.load(str(artifacts.files[name])).eval()
      for name in roles
      if name in ("rally", "standing")
    }
    self.contract = artifacts.contract or {}
    if self.contract:
      if (
        tuple(self.contract["joint_names"]) != G1_PINGPONG_JOINT_NAMES
        or abs(self.contract["control_dt"] - 0.02) > 1e-8
      ):
        raise ValueError(
          "Planner joint order/control period differs from this platform"
        )
      self.default_q = torch.tensor(self.contract["default_joints"])
      self.scales = torch.tensor(self.contract["action_scales"])
    self.generator = None
    if "mvae" in roles:
      model, _ = load_mvae(artifacts.files["mvae"])
      self.generator = MotionGenerator(model, 1, "cpu")
    self.reset()

  @torch.inference_mode()
  def reset(self, reference_time=0.0):
    self.frozen_reference = None
    self.reference.at(reference_time)
    self.reference_time = float(reference_time)
    self.tracker_delta = 0.0
    self.joint_action = torch.zeros(1, 29)
    self.high_action = torch.zeros(1, 33)
    self.alpha = torch.ones(1)
    self.latent = torch.zeros(1, 32)
    self.info: dict[str, Any] = {}
    if self.generator:
      self.generator.reset(torch.tensor([0]), self.reference.state, self.reference.root)

  @property
  def current(self):
    return self.frozen_reference or self.generator or self.reference

  @torch.inference_mode()
  def step(self, robot, ball_obs, speed=1.0, latent=None):
    if self.mode == "ball":
      self.info = {"policy": "ball_only"}
      return self.default_q[None]
    if self.generator:
      if self.planners:
        prefix = observation_prefix(
          self.generator,
          robot,
          self.default_q,
          self.high_action,
          0.02 * (self.alpha - 1),
        )
        obs = torch.cat((prefix, self.generator.history.flatten(1), ball_obs), -1)
        valid = bool(ball_obs[0, -2] > 0.5)
        role = (
          self.mode if self.mode != "hierarchy" else "rally" if valid else "standing"
        )
        if role == "standing":
          obs[:, -101:] = 0
        high = self.planners[role](obs)
        if not torch.isfinite(high).all():
          raise RuntimeError("Non-finite planner action")
        self.high_action = high
        self.alpha = decode_speed(
          high[:, -1],
          self.contract.get("min_speed", 0.5),
          self.contract.get("max_speed", 2.0),
        )
        self.latent = high[:, :-1].clamp(
          -self.contract.get("latent_clip", 3.0), self.contract.get("latent_clip", 3.0)
        )
        self.info = {"policy": role, "planner_observation": obs.clone()}
      else:
        self.alpha[:] = speed
        self.latent = (
          torch.zeros(1, 32)
          if latent is None
          else torch.as_tensor(latent, dtype=torch.float32).reshape(1, 32).clamp(-3, 3)
        )
        self.info = {"policy": self.mode}
      self.generator.step(self.latent, self.alpha)
    else:
      self.alpha[:] = speed
      self.info = {"policy": "tracker"}
    if self.mode == "mvae":
      return self.current.state[:, 12:41]
    obs = observation_prefix(
      self.current,
      robot,
      self.default_q,
      self.joint_action,
      0.02 * (self.alpha - 1) if self.generator else torch.tensor([self.tracker_delta]),
    )
    assert self.tracker is not None
    self.joint_action = self.tracker(obs)
    self.info["tracker_observation"] = obs.clone()
    target = self.default_q + self.scales * self.joint_action
    if target.shape != (1, 29) or not torch.isfinite(target).all():
      raise ValueError("Invalid tracker joint targets")
    return target

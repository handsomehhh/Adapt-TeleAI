"""Portable inference stack. A robot IO adapter supplies measured state/positions.

No hardware IO or motor loop is opened by this module. Simulation and runtime
share MotionGenerator, FK, speed mapping and position-only ball estimation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

from .actions import decode_speed
from .ball import BallPhysics, PositionEstimator, predict_trajectory
from .generator import MotionGenerator
from .mvae import file_hash, load_mvae


@dataclass
class RobotState:
  """Joint positions include the encoder bias, matching Stage-1 training inputs."""

  root_position: torch.Tensor
  root_quaternion: torch.Tensor
  joint_positions: torch.Tensor
  joint_velocities: torch.Tensor
  root_angular_velocity_body: torch.Tensor
  encoder_bias: torch.Tensor | None = None


def observation_prefix(generator, robot, default_joints, previous_action, delta_dt):
  """Exact ordered Stage-1 actor terms, with caller-selected action history."""
  bias = (
    robot.encoder_bias
    if robot.encoder_bias is not None
    else torch.zeros_like(robot.joint_positions)
  )
  positions, quaternions = generator.fk(
    robot.root_position, robot.root_quaternion, robot.joint_positions - bias
  )
  anchor = generator.fk.names.index("torso_link")
  anchor_position, anchor_quaternion = positions[:, anchor], quaternions[:, anchor]
  reference_position = generator.positions[:, anchor]
  reference_quaternion = generator.quaternions[:, anchor]
  relative_position = quat_apply_inverse(
    anchor_quaternion, reference_position - anchor_position
  )
  relative_rotation = matrix_from_quat(
    quat_mul(quat_inv(anchor_quaternion), reference_quaternion)
  )[..., :2].reshape(len(positions), 6)
  gravity = torch.zeros_like(robot.root_position)
  gravity[:, 2] = -1
  gravity = quat_apply_inverse(robot.root_quaternion, gravity)
  return torch.cat(
    (
      generator.state[:, 12:41],
      relative_position,
      relative_rotation,
      gravity,
      robot.root_angular_velocity_body,
      robot.joint_positions - default_joints,
      robot.joint_velocities,
      previous_action,
      delta_dt[:, None],
    ),
    -1,
  )


class HierarchicalController:
  def __init__(self, directory, batch=1, device="cpu"):
    root = Path(directory)
    self.info = json.loads((root / "bundle.json").read_text())
    if self.info.get("schema") != "g1-adapt-bundle-v1":
      raise ValueError("Unknown runtime bundle schema")
    if tuple(self.info["joint_names"]) != G1_PINGPONG_JOINT_NAMES:
      raise ValueError("Runtime joint order differs from native G1 FK")
    for name, digest in self.info["hashes"].items():
      if file_hash(root / name) != digest:
        raise ValueError(f"Runtime asset hash mismatch: {name}")
    model, _ = load_mvae(root / "mvae.pt", device)
    self.generator = MotionGenerator(model, batch, device, self.info["control_dt"])
    self.rally = torch.jit.load(str(root / "rally.pt"), map_location=device).eval()
    self.standing = torch.jit.load(
      str(root / "standing.pt"), map_location=device
    ).eval()
    self.tracker = torch.jit.load(str(root / "tracker.pt"), map_location=device).eval()
    self.initial = torch.load(
      root / "initial.pt", map_location=device, weights_only=False
    )
    self.default_joints = torch.tensor(self.info["default_joints"], device=device)
    self.scales = torch.tensor(self.info["action_scales"], device=device)
    self.high_action = torch.zeros(batch, model.cfg.latent_dim + 1, device=device)
    self.joint_action = torch.zeros(batch, 29, device=device)
    self.estimator = PositionEstimator(batch, device)
    self.physics = BallPhysics(**self.info["ball_physics"])
    self.batch, self.device = batch, device
    self.initialized = False

  def reset(self, robot):
    ids = torch.arange(self.batch, device=self.device)
    self.generator.reset(
      ids,
      self.initial["state"].to(self.device).expand(self.batch, -1),
      robot.root_position,
    )
    self.high_action.zero_()
    self.joint_action.zero_()
    self.estimator.reset(
      ids,
      torch.zeros(self.batch, 3, device=self.device),
      torch.zeros(self.batch, device=self.device),
    )
    self.initialized = True

  @torch.no_grad()
  def step(
    self, robot, ball_position, timestamp, target=None, measurement_timestamp=None
  ):
    if not self.initialized:
      self.reset(robot)
    now = torch.as_tensor(timestamp, device=self.device).float().expand(self.batch)
    if ball_position is None:
      received = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
      ball_position = torch.zeros(self.batch, 3, device=self.device)
    else:
      received = torch.isfinite(ball_position).all(-1)
    packet_time = (
      now
      if measurement_timestamp is None
      else torch.as_tensor(measurement_timestamp, device=self.device)
      .float()
      .expand(self.batch)
    )
    received = received & (packet_time <= now)
    self.estimator.observe(ball_position, packet_time, received)
    position, velocity, valid, age = self.estimator.estimate(now, self.physics.gravity)
    future = predict_trajectory(
      position,
      velocity,
      self.physics,
      self.info["prediction_steps"],
      self.info["control_dt"],
    )
    if target is None:
      target = torch.tensor(
        [2.7, 0, self.physics.table_height + self.physics.radius], device=self.device
      ).expand(self.batch, -1)
    ball = torch.cat(
      (
        position - robot.root_position,
        velocity,
        (future - robot.root_position[:, None]).flatten(1),
        target - robot.root_position,
        valid.float()[:, None],
        age[:, None],
      ),
      -1,
    )
    ball = torch.where(valid[:, None], ball, torch.zeros_like(ball))
    delta_dt = self.info["control_dt"] * (self.generator.last_alpha - 1)
    prefix = observation_prefix(
      self.generator, robot, self.default_joints, self.high_action, delta_dt
    )
    planner_obs = torch.cat((prefix, self.generator.history.flatten(1), ball), -1)
    if planner_obs.shape[1] != self.info["planner_observation_dim"]:
      raise ValueError("Runtime observation schema does not match planner checkpoint")
    high = torch.where(
      valid[:, None], self.rally(planner_obs), self.standing(planner_obs)
    )
    alpha = decode_speed(high[:, -1], self.info["min_speed"], self.info["max_speed"])
    self.generator.step(
      high[:, :-1].clamp(-self.info["latent_clip"], self.info["latent_clip"]), alpha
    )
    tracker_obs = observation_prefix(
      self.generator,
      robot,
      self.default_joints,
      self.joint_action,
      self.info["control_dt"] * (alpha - 1),
    )
    self.joint_action = self.tracker(tracker_obs)
    self.high_action.copy_(high)
    bias = robot.encoder_bias if robot.encoder_bias is not None else 0.0
    targets = self.joint_action * self.scales + self.default_joints - bias
    if not torch.isfinite(targets).all():
      raise RuntimeError("Non-finite hierarchical controller output")
    return targets, {
      "standing": ~valid,
      "alpha": alpha,
      "planner_observation": planner_obs,
      "tracker_observation": tracker_obs,
    }

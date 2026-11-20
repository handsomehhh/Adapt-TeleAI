"""G1 reference-time accounting and swing tracking rewards."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_RACKET_NORMAL,
)
from mjlab.utils.lab_api.math import quat_apply

from . import observations, rewards
from .commands import MotionCommand, MotionCommandCfg


class Stage1MotionCommand(MotionCommand):
  cfg: Stage1MotionCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._start_time = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completion_fraction"] = torch.zeros_like(self._start_time)
    self.metrics["reference_dt"] = torch.zeros_like(self._start_time)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self._start_time[env_ids] = self._time_seconds_progress[env_ids]
    self.last_dt[env_ids] = 0.0
    self._pending_dt[env_ids] = self._fixed_dt

  def _update_metrics(self):
    super()._update_metrics()
    end = (self.motion.lengths[self.motion_ids] - 1) / self.motion.fps_values[
      self.motion_ids
    ]
    self.metrics["completion_fraction"][:] = (
      (self._time_seconds_progress - self._start_time)
      / (end - self._start_time).clamp_min(self._fixed_dt)
    ).clamp(0, 1)
    self.metrics["reference_dt"][:] = self._fixed_dt + self.last_dt

  def reset_to_frame(self, env_ids, frame):
    super().reset_to_frame(env_ids, frame)
    self._start_time[env_ids] = self._time_seconds_progress[env_ids]
    self.last_dt[env_ids] = 0
    self._pending_dt[env_ids] = self._fixed_dt

  def reward_timeline_scale(self):
    if not self.cfg.use_reference_time_discount:
      # Physics advances at a constant control period. Random
      # reference speed is an observation/action, not a variable MDP duration.
      return torch.ones_like(self.last_dt)
    # Random dt is selected when producing the current reference, at the end
    # of the previous step. The base command has already cleared pending_dt.
    if self.cfg.random_dt_training_enabled:
      return (self._fixed_dt + self.last_dt) / self._fixed_dt
    return super().reward_timeline_scale()


@dataclass(kw_only=True)
class Stage1MotionCommandCfg(MotionCommandCfg):
  use_reference_time_discount: bool = False
  """Opt into parametric GAE in reference time instead of physical time."""

  def build(self, env):
    return Stage1MotionCommand(self, env)


def wrist_tracking(
  env,
  std: float = 0.20,
  joint_names: tuple[str, ...] = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ),
):
  return rewards.motion_joint_position_error_exp(
    env,
    command_name="motion",
    std=std,
    joint_names=joint_names,
  )


def hit_arm_tracking(
  env,
  body_names: tuple[str, ...] = ("right_elbow_link", "right_wrist_yaw_link"),
  strike_phase: float = 0.46,
):
  """Continuous tracking with extra emphasis at the dataset's strike phase."""
  phase = observations.motion_phase(env).squeeze(-1)
  emphasis = 1.0 + 4.0 * torch.exp(-(((phase - strike_phase) / 0.08) ** 2))
  reward = rewards.motion_relative_body_position_error_exp(
    env,
    command_name="motion",
    std=0.15,
    body_names=body_names,
  )
  return emphasis * reward


def racket_tracking(env, std: float, normal: bool = False):
  """Track the physical extended-racket center or its normal in world space."""
  command = env.command_manager.get_term("motion")
  reference_id = command.cfg.body_names.index("right_wrist_yaw_link")
  robot_id = command.robot.body_names.index("right_wrist_yaw_link")
  reference_quat = command.body_quat_w[:, reference_id]
  robot_quat = command.robot.data.body_link_quat_w[:, robot_id]
  if normal:
    direction = torch.tensor(G1_PINGPONG_RACKET_NORMAL, device=env.device).expand(
      env.num_envs, -1
    )
    reference = quat_apply(reference_quat, direction)
    actual = quat_apply(robot_quat, direction)
    error = (1 - (reference * actual).sum(-1)).clamp_min(0)
    denominator = 1 - torch.cos(torch.tensor(std, device=env.device))
  else:
    offset = torch.tensor(G1_PINGPONG_MOUNT_OFFSET, device=env.device).expand(
      env.num_envs, -1
    )
    reference = command.body_pos_w[:, reference_id] + quat_apply(reference_quat, offset)
    actual = command.robot.data.body_link_pos_w[:, robot_id] + quat_apply(
      robot_quat, offset
    )
    error = (reference - actual).square().sum(-1)
    denominator = std**2
  return torch.exp(-error / denominator)

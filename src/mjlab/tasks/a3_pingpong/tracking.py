"""A3-specific motion accounting and observations shared by both stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.adapt_tennis import mdp
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand, MotionCommandCfg


class A3MotionCommand(MotionCommand):
  cfg: A3MotionCommandCfg

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
      # Physics and the ball advance at a constant control period. Random
      # reference speed is an observation/action, not a variable MDP duration.
      return torch.ones_like(self.last_dt)
    # Random dt is selected when producing the current reference, at the end
    # of the previous step. The base command has already cleared pending_dt.
    if self.cfg.random_dt_training_enabled:
      return (self._fixed_dt + self.last_dt) / self._fixed_dt
    return super().reward_timeline_scale()


@dataclass(kw_only=True)
class A3MotionCommandCfg(MotionCommandCfg):
  use_reference_time_discount: bool = False
  """Opt into parametric GAE in reference time instead of physical time."""

  def build(self, env):
    return A3MotionCommand(self, env)


def joint_action_history(env):
  term = env.action_manager.get_term("joint_pos")
  return getattr(term, "joint_action", term.raw_action)


def wrist_tracking(
  env,
  std: float = 0.20,
  joint_names: tuple[str, ...] = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ),
):
  return mdp.motion_joint_position_error_exp(
    env,
    command_name="motion",
    std=std,
    joint_names=joint_names,
  )


def hit_arm_tracking(
  env,
  body_names: tuple[str, ...] = ("right_elbow_Link", "right_wrist_yaw_Link"),
  strike_phase: float = 0.46,
):
  """Continuous tracking with extra emphasis at the dataset's strike phase."""
  phase = mdp.motion_phase(env).squeeze(-1)
  emphasis = 1.0 + 4.0 * torch.exp(-(((phase - strike_phase) / 0.08) ** 2))
  reward = mdp.motion_relative_body_position_error_exp(
    env,
    command_name="motion",
    std=0.15,
    body_names=body_names,
  )
  return emphasis * reward


def residual_l2(env):
  term = env.action_manager.get_term("joint_pos")
  return term.raw_action[:, :-1].square().mean(-1)


def speed_regularization(env):
  term = env.action_manager.get_term("joint_pos")
  return (term.speed - 1.0).square()


def motion_speed(env):
  return env.action_manager.get_term("joint_pos").speed


def ball_finished(env):
  ball = env.command_manager.get_term("ball")
  return ball.missed | ball.landed

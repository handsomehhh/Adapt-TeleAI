"""Ordered A3 joint actions and a frozen-tracker residual controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand


class OrderedJointPositionAction(JointPositionAction):
  def _find_targets(self, cfg):
    return self._entity.find_joints(cfg.actuator_names, preserve_order=True)


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env):
    return OrderedJointPositionAction(self, env)


class FrozenTrackerAction(OrderedJointPositionAction):
  """Joint residuals are in tracker-action units; final component controls speed.

  The tracker is a Stage-1 JIT policy with frozen normalization. It receives
  exactly the same ordered observation terms as Stage 1. Only the high-level
  receiving policy is optimized in Stage 2.
  """

  cfg: FrozenTrackerActionCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    path = Path(cfg.tracker_file).expanduser()
    if not path.is_file():
      raise FileNotFoundError(
        f"Stage 2 requires a trained Stage-1 JIT tracker: {path}. "
        "Set --env.actions.joint-pos.tracker-file to jit/modeljit_N.pt."
      )
    self.tracker = torch.jit.load(str(path), map_location=self.device).eval()
    for parameter in self.tracker.parameters():
      parameter.requires_grad_(False)
    self._action_dim = self._num_targets + 1
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self.joint_action = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    self.speed = torch.ones(self.num_envs, device=self.device)
    self._residual_scales = torch.tensor(
      [
        cfg.hit_arm_scale
        if name.startswith(cfg.hit_arm_joint_prefixes)
        else cfg.residual_scale
        for name in self._target_names
      ],
      device=self.device,
    )

  def process_actions(self, actions):
    self._raw_actions.copy_(actions)
    u = actions[:, -1].clamp(-1, 1)
    self.speed = torch.where(
      u >= 0,
      1 + u * (self.cfg.max_speed - 1),
      1 + u * (1 - self.cfg.min_speed),
    )
    motion = cast(MotionCommand, self._env.command_manager.get_term("motion"))
    obs = self._env.observation_manager.compute_group("tracker")
    with torch.no_grad():
      base_action = self.tracker(obs)
    # The tracker sees the speed that produced its current reference, just as
    # in Stage 1. The high-level action selects the NEXT reference advance.
    motion.set_pending_dt(motion.cfg.fixed_dt * self.speed)
    residual = actions[:, :-1].clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
    self.joint_action[:] = base_action + self._residual_scales * residual
    self._processed_actions = self.joint_action * self._scale + self._offset

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if hasattr(self, "joint_action"):
      self.joint_action[env_ids] = 0
      self.speed[env_ids] = 1


@dataclass(kw_only=True)
class FrozenTrackerActionCfg(OrderedJointPositionActionCfg):
  tracker_file: str = "ckpts/a3_pingpong/tracker.pt"
  residual_scale: float = 0.25
  hit_arm_scale: float = 0.25
  """Independent residual authority on the seven hitting-arm joints."""
  hit_arm_joint_prefixes: tuple[str, ...] = (
    "right_shoulder_",
    "right_elbow_",
    "right_wrist_",
  )
  residual_clip: float = 2.0
  min_speed: float = 0.5
  max_speed: float = 2.0

  def build(self, env):
    return FrozenTrackerAction(self, env)

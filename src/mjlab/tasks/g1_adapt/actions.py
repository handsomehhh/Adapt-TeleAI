"""Planner actions are z + speed; the frozen tracker alone outputs joint actions."""

from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.tasks.g1_stage1.actions import (
  OrderedJointPositionAction,
  OrderedJointPositionActionCfg,
)

from .reference import GeneratedMotionCommand


def tracker_action_history(env):
  return env.action_manager.get_term("joint_pos").joint_action


def decode_speed(value, minimum=0.5, maximum=2.0):
  if not 0 < minimum <= 1 <= maximum:
    raise ValueError("Speed bounds must contain 1 and be positive")
  value = value.clamp(-1, 1)
  return torch.where(value >= 0, 1 + value * (maximum - 1), 1 + value * (1 - minimum))


class LatentTrackerAction(OrderedJointPositionAction):
  cfg: "LatentTrackerActionCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if not Path(cfg.tracker_file).is_file():
      raise FileNotFoundError(
        f"Train Stage 1 and set ADAPT_G1_TRACKER or --env.actions.joint-pos.tracker-file: {cfg.tracker_file}"
      )
    self.tracker = torch.jit.load(cfg.tracker_file, map_location=self.device).eval()
    for parameter in self.tracker.parameters():
      parameter.requires_grad_(False)
    model = env.command_manager.get_term("motion").generator.model
    if model.cfg.latent_dim != cfg.latent_dim:
      raise ValueError("Planner latent_dim must match the frozen MVAE checkpoint")
    self._action_dim = cfg.latent_dim + 1
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self.joint_action = torch.zeros(self.num_envs, 29, device=self.device)
    self.speed = torch.ones(self.num_envs, device=self.device)

  @torch.no_grad()
  def process_actions(self, actions):
    self._raw_actions.copy_(actions)
    self.speed = decode_speed(actions[:, -1], self.cfg.min_speed, self.cfg.max_speed)
    motion = self._env.command_manager.get_term("motion")
    assert isinstance(motion, GeneratedMotionCommand)
    motion.advance(
      actions[:, :-1].clamp(-self.cfg.latent_clip, self.cfg.latent_clip), self.speed
    )
    tracker_obs = self._env.observation_manager.compute_group("tracker")
    self.joint_action = self.tracker(tracker_obs)
    if (
      self.joint_action.shape != (self.num_envs, 29)
      or not torch.isfinite(self.joint_action).all()
    ):
      raise RuntimeError("Frozen tracker output or observation contract is invalid")
    self._processed_actions = self.joint_action * self._scale + self._offset

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if hasattr(self, "joint_action"):
      self.joint_action[env_ids] = 0
      self.speed[env_ids] = 1


@dataclass(kw_only=True)
class LatentTrackerActionCfg(OrderedJointPositionActionCfg):
  tracker_file: str = "artifacts/g1_adapt/tracker.pt"
  latent_dim: int = 32
  min_speed: float = 0.5
  max_speed: float = 2.0
  latent_clip: float = 3.0

  def build(self, env):
    return LatentTrackerAction(self, env)

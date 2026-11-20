"""Replace offline reference playback with generated references after reset."""

from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.tasks.g1_stage1.mdp.commands import MotionCommand
from mjlab.tasks.g1_stage1.mdp.tracking import (
  Stage1MotionCommand,
  Stage1MotionCommandCfg,
)

from .generator import MotionGenerator
from .motion import PHASE, pack_state
from .mvae import load_mvae


class GeneratedMotionCommand(Stage1MotionCommand):
  def __init__(self, cfg, env):
    self.generator = None
    if not Path(cfg.mvae_file).is_file():
      raise FileNotFoundError(
        f"Train MVAE first and set ADAPT_G1_MVAE or --env.commands.motion.mvae-file: {cfg.mvae_file}"
      )
    super().__init__(cfg, env)
    model, self.mvae_metadata = load_mvae(cfg.mvae_file, self.device)
    self.generator = MotionGenerator(model, self.num_envs, self.device, env.step_dt)
    self.generated_body_ids = torch.tensor(
      [self.generator.fk.names.index(n) for n in cfg.body_names], device=self.device
    )
    self._initialize_generated(torch.arange(self.num_envs, device=self.device))

  def _initialize_generated(self, ids):
    assert self.generator is not None
    library = self.motion
    motion_ids, frames = self.motion_ids[ids], self.time_steps[ids]
    pos = library._gather("body_pos_w", motion_ids, frames)
    quat = library._gather("body_quat_w", motion_ids, frames)
    length = (library.lengths[motion_ids] - 1).clamp_min(1)
    phase = frames.float() / length * (2 * torch.pi)
    state = pack_state(
      library._gather("joint_pos", motion_ids, frames),
      library._gather("joint_vel", motion_ids, frames),
      quat[:, 0],
      library._gather("body_lin_vel_w", motion_ids, frames)[:, 0],
      library._gather("body_ang_vel_w", motion_ids, frames)[:, 0],
      torch.stack((phase.cos(), phase.sin()), -1),
    )
    if not hasattr(self, "ready_joints"):
      self.ready_joints = torch.zeros(self.num_envs, 29, device=self.device)
      self.ready_root = torch.zeros(self.num_envs, 3, device=self.device)
    self.ready_joints[ids] = state[:, 12:41]
    self.ready_root[ids] = pos[:, 0] + self._env.scene.env_origins[ids]
    self.generator.reset(ids, state, pos[:, 0] + self._env.scene.env_origins[ids])
    self.update_relative_body_poses()

  def _gather_attr(self, attr):
    if self.generator is None:
      return super()._gather_attr(attr)
    if attr == "joint_pos":
      return self.generator.state[:, 12:41]
    if attr == "joint_vel":
      return self.generator.joint_velocities
    fields = {
      "body_pos_w": self.generator.positions,
      "body_quat_w": self.generator.quaternions,
      "body_lin_vel_w": self.generator.linear_velocities,
      "body_ang_vel_w": self.generator.angular_velocities,
    }
    value = fields[attr][:, self.generated_body_ids]
    if attr == "body_pos_w":
      value = value - self._env.scene.env_origins[:, None]
    return value

  def _resample_command(self, env_ids):
    generator = self.generator
    self.generator = None
    try:
      super()._resample_command(env_ids)
    finally:
      self.generator = generator
    if generator is not None:
      self._initialize_generated(env_ids)

  def _update_command(self):
    # Actions produce references before the physics step; never advance again here.
    self.update_relative_body_poses()

  def _update_metrics(self):
    MotionCommand._update_metrics(self)
    if self.generator is not None:
      self.metrics["reference_dt"][:] = self._fixed_dt * self.generator.last_alpha

  def advance(self, latent, alpha):
    assert self.generator is not None
    self.generator.step(latent, alpha)
    self.last_dt[:] = self._fixed_dt * (alpha - 1)
    self._time_seconds_progress += self._fixed_dt * alpha
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class GeneratedMotionCommandCfg(Stage1MotionCommandCfg):
  mvae_file: str = "artifacts/g1_adapt/mvae.pt"

  def build(self, env):
    return GeneratedMotionCommand(self, env)


def generated_history(env):
  motion = env.command_manager.get_term("motion")
  return motion.generator.history.flatten(1)


def generated_phase(env):
  return env.command_manager.get_term("motion").generator.state[:, PHASE]

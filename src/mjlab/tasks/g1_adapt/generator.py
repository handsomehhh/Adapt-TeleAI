"""Autoregressive generation + speed adaptation, shared by simulator and runtime."""

import torch

from .motion import (
  JOINT_VELOCITIES,
  JOINTS,
  G1Kinematics,
  angular_velocity,
  blend_state,
  quaternion_6d,
)


class MotionGenerator:
  def __init__(self, model, num_envs, device, dt=0.02):
    self.model = model.eval().requires_grad_(False)
    self.fk = G1Kinematics(device)
    self.dt = dt
    if abs(model.cfg.fps * dt - 1) > 1e-5:
      raise ValueError("MVAE frame rate must match the fixed control rate")
    self.history = torch.zeros(
      num_envs, model.cfg.history, model.cfg.state_dim, device=device
    )
    self.root = torch.zeros(num_envs, 3, device=device)
    self.positions = torch.zeros(num_envs, len(self.fk.names), 3, device=device)
    self.quaternions = torch.zeros(num_envs, len(self.fk.names), 4, device=device)
    self.quaternions[..., 0] = 1
    self.linear_velocities = torch.zeros_like(self.positions)
    self.angular_velocities = torch.zeros_like(self.positions)
    self.joint_velocities = torch.zeros(num_envs, 29, device=device)
    self.stroke_logits = torch.zeros(num_envs, 2, device=device)
    self.spin_logits = torch.zeros_like(self.stroke_logits)
    self.last_alpha = torch.ones(num_envs, device=device)

  @property
  def state(self):
    return self.history[:, -1]

  def reset(self, ids, state, root):
    self.history[ids] = state[:, None]
    self.root[ids] = root
    positions, quaternions = self.fk(
      root, quaternion_6d(state[:, 6:12]), state[:, JOINTS]
    )
    self.positions[ids], self.quaternions[ids] = positions, quaternions
    self.linear_velocities[ids] = 0
    self.angular_velocities[ids] = 0
    self.joint_velocities[ids] = state[:, JOINT_VELOCITIES]
    self.last_alpha[ids] = 1

  @torch.no_grad()
  def step(self, latent, alpha):
    if latent.shape != (len(self.history), self.model.cfg.latent_dim):
      raise ValueError("Latent action shape does not match the MVAE")
    if (
      not torch.isfinite(latent).all()
      or not torch.isfinite(alpha).all()
      or (alpha <= 0).any()
    ):
      raise ValueError("Invalid latent/speed command")
    proposed, self.stroke_logits, self.spin_logits = self.model.decode(
      self.history, latent
    )
    state = blend_state(self.state, proposed, alpha)
    if not torch.isfinite(state).all():
      raise RuntimeError("Generated reference became non-finite")
    # Safety bounds are explicit deployment choices, not additional joint residual actions.
    state[:, JOINTS] = torch.clamp(
      state[:, JOINTS], self.fk.limits[:, 0], self.fk.limits[:, 1]
    )
    state[:, :3] = state[:, :3].clamp(-3, 3)
    root = self.root + state[:, :3] * alpha[:, None] * self.dt
    positions, quaternions = self.fk(
      root, quaternion_6d(state[:, 6:12]), state[:, JOINTS]
    )
    self.linear_velocities = (positions - self.positions) / self.dt
    self.angular_velocities = angular_velocity(self.quaternions, quaternions, self.dt)
    self.joint_velocities = (state[:, JOINTS] - self.state[:, JOINTS]) / self.dt
    self.positions, self.quaternions, self.root = positions, quaternions, root
    # Critical AdaPT detail: feed the speed-adapted state back, not the raw decoder proposal.
    self.history = torch.cat((self.history[:, 1:], state[:, None]), 1)
    self.last_alpha.copy_(alpha)
    return state

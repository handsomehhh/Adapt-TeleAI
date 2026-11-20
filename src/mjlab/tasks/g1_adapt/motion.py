"""Robot motion representation and batched FK shared by training and inference.

The MVAE state deliberately excludes absolute root position. Rotation uses the
first two *columns* of a rotation matrix, flattened column first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
  get_spec,
)
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv, quat_mul

STATE_DIM = 72
JOINTS = slice(12, 41)
JOINT_VELOCITIES = slice(41, 70)
PHASE = slice(70, 72)
SCHEMA = "g1-adapt-motion-v1"


def rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quaternion)[..., :, :2].transpose(-1, -2).flatten(-2)


def matrix_6d(value: torch.Tensor) -> torch.Tensor:
  first, second = value[..., :3], value[..., 3:]
  default = torch.zeros_like(first)
  default[..., 0] = 1
  first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-7, first, default)
  first = F.normalize(first, dim=-1)
  second = second - (first * second).sum(-1, keepdim=True) * first
  fallback = torch.zeros_like(second)
  fallback[..., 1] = 1
  alternate = torch.zeros_like(second)
  alternate[..., 2] = 1
  fallback = torch.where(first[..., 1:2].abs() < 0.9, fallback, alternate)
  fallback = fallback - (fallback * first).sum(-1, keepdim=True) * first
  second = F.normalize(
    torch.where(second.norm(dim=-1, keepdim=True) > 1e-7, second, fallback), dim=-1
  )
  return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def matrix_quaternion(matrix: torch.Tensor) -> torch.Tensor:
  """Stable matrix -> WXYZ, including 180-degree rotations."""
  m = matrix
  candidates = torch.stack(
    (
      torch.stack(
        (
          1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
          m[..., 2, 1] - m[..., 1, 2],
          m[..., 0, 2] - m[..., 2, 0],
          m[..., 1, 0] - m[..., 0, 1],
        ),
        -1,
      ),
      torch.stack(
        (
          m[..., 2, 1] - m[..., 1, 2],
          1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
          m[..., 0, 1] + m[..., 1, 0],
          m[..., 0, 2] + m[..., 2, 0],
        ),
        -1,
      ),
      torch.stack(
        (
          m[..., 0, 2] - m[..., 2, 0],
          m[..., 0, 1] + m[..., 1, 0],
          1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
          m[..., 1, 2] + m[..., 2, 1],
        ),
        -1,
      ),
      torch.stack(
        (
          m[..., 1, 0] - m[..., 0, 1],
          m[..., 0, 2] + m[..., 2, 0],
          m[..., 1, 2] + m[..., 2, 1],
          1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
        -1,
      ),
    ),
    -2,
  )
  index = candidates.diagonal(dim1=-2, dim2=-1).argmax(-1)
  chosen = candidates.gather(
    -2, index[..., None, None].expand(*index.shape, 1, 4)
  ).squeeze(-2)
  return F.normalize(chosen, dim=-1)


def quaternion_6d(value: torch.Tensor) -> torch.Tensor:
  return matrix_quaternion(matrix_6d(value))


def pack_state(joints, velocities, root_quat, root_vel, root_ang, phase):
  return torch.cat(
    (root_vel, root_ang, rotation_6d(root_quat), joints, velocities, phase), -1
  )


def blend_state(previous, proposed, alpha):
  """Interpolate/extrapolate a motion state; return the state fed back to MVAE."""
  mixed = previous + alpha[..., None] * (proposed - previous)
  rotation = matrix_6d(mixed[..., 6:12])[..., :, :2].transpose(-1, -2).flatten(-2)
  phase = mixed[..., PHASE]
  fallback = torch.zeros_like(phase)
  fallback[..., 0] = 1
  phase = F.normalize(
    torch.where(phase.norm(dim=-1, keepdim=True) > 1e-7, phase, fallback), dim=-1
  )
  return torch.cat((mixed[..., :6], rotation, mixed[..., 12:70], phase), -1)


class G1Kinematics:
  """Torch FK from the exact MJCF, including nonzero joint pivots and qpos0."""

  def __init__(self, device="cpu"):
    model = get_spec().compile()
    self.names = tuple(model.body(i).name for i in range(1, model.nbody))
    self.parents = model.body_parentid[1:].tolist()
    self.joints = {}
    self.device = device
    self.pos = torch.as_tensor(
      model.body_pos[1:].copy(), dtype=torch.float32, device=device
    )
    self.quat = torch.as_tensor(
      model.body_quat[1:].copy(), dtype=torch.float32, device=device
    )
    limits = []
    for column, name in enumerate(G1_PINGPONG_JOINT_NAMES):
      joint = model.joint(name)
      body = int(joint.bodyid[0]) - 1
      self.joints[body] = (
        column,
        torch.as_tensor(joint.axis.copy(), dtype=torch.float32, device=device),
        torch.as_tensor(joint.pos.copy(), dtype=torch.float32, device=device),
        float(model.qpos0[int(joint.qposadr[0])]),
      )
      limits.append(joint.range.tolist())
    self.limits = torch.tensor(limits, dtype=torch.float32, device=device)

  def __call__(self, root_position, root_quaternion, joint_positions):
    count = joint_positions.shape[0]
    positions, quaternions = [root_position], [root_quaternion]
    for body in range(1, len(self.names)):
      parent = self.parents[body] - 1
      local_pos = self.pos[body].expand(count, -1)
      local_quat = self.quat[body].expand(count, -1)
      if body in self.joints:
        column, axis, pivot, reference = self.joints[body]
        half = (joint_positions[:, column] - reference) * 0.5
        rotation = torch.cat((half.cos()[:, None], half.sin()[:, None] * axis), -1)
        local_pos = local_pos + quat_apply(
          local_quat,
          pivot.expand(count, -1) - quat_apply(rotation, pivot.expand(count, -1)),
        )
        local_quat = quat_mul(local_quat, rotation)
      positions.append(positions[parent] + quat_apply(quaternions[parent], local_pos))
      quaternions.append(quat_mul(quaternions[parent], local_quat))
    return torch.stack(positions, 1), torch.stack(quaternions, 1)


def angular_velocity(previous, current, dt):
  delta = quat_mul(current, quat_inv(previous))
  delta = torch.where(delta[..., :1] < 0, -delta, delta)
  norm = delta[..., 1:].norm(dim=-1, keepdim=True)
  return (
    delta[..., 1:] * (2 * torch.atan2(norm, delta[..., :1]) / norm.clamp_min(1e-7)) / dt
  )

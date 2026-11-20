"""HOPE-style motion catalogue and torso-aligned reference views (Python)."""

import json
from pathlib import Path

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import Motion, quaternion_multiply, rotation, slerp


def heading(quaternion):
  """World yaw of a wxyz quaternion's x axis."""
  r = rotation(quaternion)
  return np.arctan2(r[1, 0], r[0, 0])


class RebasedMotion:
  """Map a clip's first anchor to the measured anchor without changing joints."""

  def __init__(self, motion, position, quaternion):
    self.source = motion
    _, p0, q0 = motion.sample(0)
    yaw = heading(quaternion) - heading(q0)
    self.quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    self.rotation = rotation(self.quaternion)
    self.translation = position - self.rotation @ p0

  def __getattr__(self, name):
    return getattr(self.source, name)

  def sample(self, t):
    joints, position, quaternion = self.source.sample(t)
    return (
      joints,
      self.rotation @ position + self.translation,
      quaternion_multiply(self.quaternion, quaternion),
    )

  def strike_position(self, cfg):
    return self.rotation @ self.source.strike_position(cfg) + self.translation

  def strike_time(self, cfg):
    return self.source.strike_time(cfg)

  def body_position(self, name, t):
    return self.rotation @ self.source.body_position(name, t) + self.translation


class MotionLibrary:
  def __init__(self, deployment, allow=None):
    cfg = deployment.settings
    self.motions = [
      Motion(path, cfg["robot"]["joint_names"], cfg["anchor"])
      for path in deployment.motions
    ]
    self.paths = deployment.motions
    self.allow = list(range(len(self.motions))) if allow is None else list(allow)
    if (
      not self.allow
      or len(set(self.allow)) != len(self.allow)
      or any(index < 0 or index >= len(self.motions) for index in self.allow)
    ):
      raise ValueError("motion-allow must contain unique valid motion indices")
    mapping = {}
    if self.paths:
      path = self.paths[0].parent.parent / "hit_type.json"
      if path.exists():
        mapping = json.loads(path.read_text())
    self.hit_types = [mapping.get(Path(path).stem, "unknown") for path in self.paths]

  def rebase(self, index, state):
    return RebasedMotion(self.motions[index], state.anchor_pos, state.anchor_quat)


class ReturnReference:
  """Blend from measured end posture to the fixed home, as in HOPE return."""

  def __init__(self, state, home):
    self.start = (state.q.copy(), state.anchor_pos.copy(), state.anchor_quat.copy())
    self.finish = home.sample(0)
    self.alpha = 0.0

  def sample(self, t):
    q0, p0, r0 = self.start
    q1, p1, r1 = self.finish
    s = self.alpha * self.alpha * (3 - 2 * self.alpha)
    return q0 * (1 - s) + q1 * s, p0 * (1 - s) + p1 * s, slerp(r0, r1, s)

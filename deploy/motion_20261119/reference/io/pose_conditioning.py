"""Source-time pose/velocity conditioning matching A3 HopeBaseVelocityTracker.

Optional for existing tasks; configured only by the independent HOPE deployment.
"""

import numpy as np


class PoseConditioner:
  def __init__(self, cfg):
    self.cfg = cfg
    for key, default in (("alpha", 0.5), ("velocity_alpha", 0.3)):
      if not 0 < cfg.get(key, default) <= 1:
        raise ValueError(f"Pose conditioning {key} must be in (0, 1]")
    self.position = self.quaternion = self.previous_position = None
    self.source = self.arrival = None
    self.last_seen = None
    self.reject_streak = 0
    self.velocity = np.zeros(3)

  def update(self, position, quaternion, source, arrival):
    if self.source is not None and source <= self.source:
      return None
    gap = np.inf if self.last_seen is None else arrival - self.last_seen
    self.last_seen = arrival
    reseed = self.position is None
    if self.position is not None:
      distance = np.linalg.norm(position - self.position)
      if distance > self.cfg.get("jump_max_m", 0.15) > 0 and gap <= self.cfg.get(
        "jump_bypass_seconds", 0.2
      ):
        self.reject_streak += 1
        threshold = self.cfg.get("reseed_after", 10)
        if threshold <= 0 or self.reject_streak < threshold:
          return None
        reseed = True
    self.reject_streak = 0
    if reseed:
      self.position, self.quaternion = (
        position.copy(),
        quaternion / np.linalg.norm(quaternion),
      )
      self.velocity[:] = 0
    else:
      a = self.cfg.get("alpha", 0.5)
      self.position = a * position + (1 - a) * self.position
      q = quaternion if self.quaternion @ quaternion >= 0 else -quaternion
      self.quaternion = (1 - a) * self.quaternion + a * q
      self.quaternion /= np.linalg.norm(self.quaternion)
      dt = source - self.source
      self.velocity[:] = 0 if not 1e-4 < dt < 0.5 else self.velocity
      if 1e-4 < dt < 0.5:
        raw = (self.position - self.previous_position) / dt
        maximum = self.cfg.get("velocity_max_mps", 2.0)
        speed = np.linalg.norm(raw)
        if maximum > 0 and speed > maximum:
          raw *= maximum / speed
        a = self.cfg.get("velocity_alpha", 0.3)
        self.velocity = a * raw + (1 - a) * self.velocity
    self.previous_position = self.position.copy()
    self.source, self.arrival = source, arrival
    return self.position.copy(), self.quaternion.copy()

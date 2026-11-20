"""Detect rapid alternating joint velocity without altering observations."""

from collections import deque

import numpy as np


class OscillationGuard:
  def __init__(self, names, config, dt):
    self.enabled = config.get("enabled", False)
    self.indices = list(range(len(names)))
    self.names = [names[i] for i in self.indices]
    self.window = config.get("window_seconds", 0.24)
    self.speed = config.get("rms_speed", 0.8)
    self.ratio = config.get("sign_change_ratio", 0.75)
    if not np.isfinite(dt) or dt <= 0:
      raise ValueError("Invalid oscillation guard control period")
    self.minimum = max(8, int(np.ceil(0.14 / dt)))
    if (
      not isinstance(self.enabled, bool)
      or not all(np.isfinite(v) for v in (self.window, self.speed, self.ratio, dt))
      or dt <= 0
      or self.window < self.minimum * dt
      or self.speed <= 0
      or not 0 < self.ratio <= 1
    ):
      raise ValueError("Invalid oscillation guard configuration")
    self.samples = deque()
    self.debug = {}

  def reset(self):
    self.samples.clear()
    self.debug = {}

  def check(self, dq, now):
    if not self.enabled:
      return
    self.samples.append((now, np.asarray(dq)[self.indices].copy()))
    while self.samples and now - self.samples[0][0] > self.window:
      self.samples.popleft()
    if len(self.samples) < self.minimum:
      return
    t = np.array([v[0] for v in self.samples])
    velocities = np.array([v[1] for v in self.samples])
    if np.max(np.diff(t)) > self.window / 3:
      self.reset()
      return
    rms = np.sqrt(np.mean(velocities**2, axis=0))
    flips = np.mean(velocities[1:] * velocities[:-1] < 0, axis=0)
    self.debug = {
      "window_seconds": float(t[-1] - t[0]),
      "joints": {
        n: {"rms_speed": float(r), "sign_change_ratio": float(f)}
        for n, r, f in zip(self.names, rms, flips, strict=True)
      },
    }
    bad = np.flatnonzero((rms >= self.speed) & (flips >= self.ratio))
    if len(bad):
      raise ValueError(f"Rapid joint oscillation detected: {self.names[bad[0]]}")

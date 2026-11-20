"""Shared posture transition and swing lifecycle for live sim and real IO."""

import numpy as np


class Lifecycle:
  def __init__(self, initial_q, transition_seconds, *, initial_state="WAIT"):
    self.initial_q = np.asarray(initial_q, dtype=float)
    if self.initial_q.shape != (29,) or not np.isfinite(self.initial_q).all():
      raise ValueError("Invalid initial posture")
    if not np.isfinite(transition_seconds) or transition_seconds <= 0:
      raise ValueError("transition_seconds must be positive")
    if initial_state not in ("WAIT", "ARMED"):
      raise ValueError("Invalid initial lifecycle state")
    self.transition_seconds = transition_seconds
    self.state = initial_state
    self.previous_buttons = 0
    self.transition_from = None
    self.transition_start = None
    self.swing_start = None

  def edges(self, buttons):
    pressed = buttons & ~self.previous_buttons
    self.previous_buttons = buttons
    return pressed

  def prepare(self, q, now):
    if self.state != "WAIT":
      raise RuntimeError("prepare requires WAIT")
    self._transition(q, now, "PREPARE")

  def position(self, q, now):
    """Real hardware startup: open-loop reference, closed-loop joint PD."""
    if self.state != "WAIT":
      raise RuntimeError("position requires WAIT")
    self._transition(q, now, "POSITIONING")

  def posture_target(self, now):
    if self.state not in ("POSITIONING", "POSE_HOLD"):
      raise RuntimeError("posture_target requires POSITIONING or POSE_HOLD")
    alpha = float(
      np.clip((now - self.transition_start) / self.transition_seconds, 0, 1)
    )
    if alpha >= 1:
      self.state = "POSE_HOLD"
    blend = alpha * alpha * (3 - 2 * alpha)
    return self.transition_from + blend * (self.initial_q - self.transition_from)

  def arm(self, now, receiving):
    if self.state != "READY":
      raise RuntimeError("arm requires READY")
    self.state = "ARMED" if receiving else "EXECUTE"
    self.swing_start = now if not receiving else None

  def execute(self, now):
    if self.state != "ARMED":
      raise RuntimeError("execute requires ARMED")
    self.state, self.swing_start = "EXECUTE", now

  def recover(self, q, now):
    if self.state != "EXECUTE":
      raise RuntimeError("recover requires EXECUTE")
    self._transition(q, now, "RECOVER")

  def _transition(self, q, now, state):
    q = np.asarray(q, dtype=float)
    if q.shape != (29,) or not np.isfinite(q).all():
      raise ValueError("Invalid transition posture")
    self.transition_from, self.transition_start, self.state = q.copy(), now, state

  def ready(self):
    """Called when the configured measured or timed preparation is complete."""
    if self.state not in ("PREPARE", "RECOVER"):
      raise RuntimeError("ready requires PREPARE or RECOVER")
    self.state = "READY"

  def restabilize(self, q, now):
    if self.state not in ("READY", "ARMED"):
      raise RuntimeError("restabilize requires READY or ARMED")
    self._transition(q, now, "PREPARE")

  def halt(self):
    self.state = "HALT"

"""Real-only stop behavior matching public HOPE-TeleAI main at 67a1a72.

G1 transport/mode watchdogs remain necessary: its independent writer must not
repeat an abandoned position target indefinitely. MuJoCo keeps its old profile.
"""

import copy

import numpy as np

PROFILE = "hope_67a1a72"


def real_stop_settings(config):
  profile = config.get("stop_policy", "legacy")
  if profile == "legacy":
    return copy.deepcopy(config)
  if profile != PROFILE:
    raise ValueError(f"Unknown real stop_policy: {profile}")
  cfg = copy.deepcopy(config)
  cfg.update(
    state_timeout=0.3,
    command_timeout=0.3,
    max_control_overrun=None,
    max_tilt_degrees=35.0,
    latch_damping_on_fault=True,
    halt_with_policy_kd=True,
    root_fresh_seconds=0.2,
    root_grace_seconds=0.5,
    continue_swing_without_ball=True,
  )
  cfg["guard"].update(
    enforce_measured_limits=False,
    enforce_measured_velocity=False,
    enforce_measured_step=False,
  )
  cfg.setdefault("stable_policy", {})["preposition_timeout"] = None
  cfg["stable_policy"]["handoff_timeout"] = None
  return cfg


class RootPoseGrace:
  """Freeze an already received pose through HOPE's 0.2s + 0.5s window."""

  def __init__(self, fresh_seconds=0.2, grace_seconds=0.5):
    self.fresh_seconds = fresh_seconds
    self.grace_seconds = grace_seconds
    self.last = None
    self.frozen = False

  def resolve(self, sample, now):
    if sample is not None:
      self.last = sample
    self.frozen = False
    if self.last is None:
      return None
    age = max(0.0, now - self.last[3])
    if age > self.fresh_seconds + self.grace_seconds:
      return None
    self.frozen = age > self.fresh_seconds
    return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in self.last)


def wait_in_damping(robot, clock, started, duration, record):
  """Keep the IO writer alive, latched in damping, until operator stop or expiry.

  Never send another policy command or automatically rearm. A failed transport
  cannot guarantee motor delivery; Ctrl-C/SIGTERM still terminate this wait.
  """
  robot.halt()
  while clock.monotonic() - started < duration:
    error = None
    try:
      feedback = robot.feedback()
    except Exception as exc:
      feedback = getattr(robot, "latest", None)
      error = str(exc)
    if feedback is not None and feedback[3] & ((1 << 3) | (1 << 9) | (1 << 10)):
      return "safety_halt_operator"
    record(feedback, error)
    clock.sleep(0.02)
  return "safety_halt_duration"

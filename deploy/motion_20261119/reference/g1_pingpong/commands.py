"""Frozen-frame command conditioning for preparation, holding and recovery."""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.pd_gains import deployment_gains


class CommandLimiter:
  def __init__(self, limits, guard, dt):
    self.limits = np.asarray(limits)
    self.guard = guard
    self.tracking = guard["max_tracking_error"] * 0.98
    self.step = dt * guard["max_target_velocity"] * 0.96

  def apply(self, desired, q, previous=None, *, preparation=True):
    if not all(
      np.asarray(a).shape == (29,) and np.isfinite(a).all() for a in (desired, q)
    ):
      raise ValueError("Invalid joint command/feedback for limiter")
    previous = q if previous is None else previous
    lower, upper = np.full(29, -np.inf), np.full(29, np.inf)
    if preparation or self.guard.get("enforce_target_limits", True):
      lower = np.maximum(lower, self.limits[:, 0])
      upper = np.minimum(upper, self.limits[:, 1])
    if preparation or self.guard.get("enforce_target_slew", True):
      lower = np.maximum(lower, previous - self.step)
      upper = np.minimum(upper, previous + self.step)
    if preparation or self.guard.get("enforce_target_tracking", True):
      lower = np.maximum(lower, q - self.tracking)
      upper = np.minimum(upper, q + self.tracking)
    if np.any(lower > upper):
      raise ValueError("No joint command satisfies the guards")
    return np.clip(desired, lower, upper)


class CommandGuard:
  def __init__(self, limits, cfg, joint_names=None):
    self.limits = np.asarray(limits)
    self.cfg = cfg
    self.joint_names = joint_names
    self.previous = None
    self.previous_q = None

  def check(self, target, q, dq, dt):
    if not all(
      np.asarray(a).shape == (29,) and np.isfinite(a).all() for a in (target, q, dq)
    ):
      raise ValueError("Invalid motor feedback/command")
    if self.cfg.get("enforce_target_limits", True) and (
      np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1])
    ):
      violation = np.maximum(self.limits[:, 0] - target, target - self.limits[:, 1])
      index = int(np.argmax(violation))
      raise ValueError(
        f"Joint target outside hardware/model limits: joint {index}, "
        f"target={target[index]:.6f}, limits={self.limits[index].tolist()}"
      )
    if self.cfg.get("enforce_target_tracking", True) and (
      np.max(np.abs(target - q)) > self.cfg["max_tracking_error"]
    ):
      index = int(np.argmax(np.abs(target - q)))
      raise ValueError(
        f"Motor tracking error exceeds limit: joint {index}, "
        f"error={abs(target[index] - q[index]):.6f} rad"
      )
    if self.cfg.get("enforce_measured_velocity", True) and (
      np.max(np.abs(dq)) > self.cfg["max_measured_velocity"]
    ):
      raise ValueError("Measured motor velocity exceeds limit")
    if (
      self.cfg.get("enforce_target_slew", True)
      and self.previous is not None
      and np.max(np.abs(target - self.previous)) / dt > self.cfg["max_target_velocity"]
    ):
      raise ValueError("Motor target slew exceeds limit")
    if (
      self.cfg.get("enforce_measured_limits", True)
      and "measured_limit_margin" in self.cfg
    ):
      margin = self.cfg["measured_limit_margin"]
      if np.any(q < self.limits[:, 0] - margin) or np.any(
        q > self.limits[:, 1] + margin
      ):
        violation = np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1])
        index = int(np.argmax(violation))
        joint = (
          f"{index} ({self.joint_names[index]})" if self.joint_names else str(index)
        )
        raise ValueError(
          f"Measured joint position exceeds configured limits: joint {joint}, "
          f"measured={q[index]:.6f}, limits={self.limits[index].tolist()}, "
          f"margin={margin:.6f}"
        )
    if (
      self.previous_q is not None
      and self.cfg.get("enforce_measured_step", True)
      and "max_measured_step" in self.cfg
      and np.max(np.abs(q - self.previous_q)) > self.cfg["max_measured_step"]
    ):
      raise ValueError("Measured joint position jumped beyond configured limit")
    self.previous = target.copy()
    self.previous_q = q.copy()


def policy_pd_settings(settings, runtime):
  """Use explicit deployment PD, retaining training action scales unchanged."""
  kp, kd = deployment_gains(
    settings["robot"]["joint_names"], scales=runtime.get("policy_pd", {})
  )
  return {
    **settings,
    "robot": {
      **settings["robot"],
      "kp": kp.tolist(),
      "kd": kd.tolist(),
      "pd_source": "deploy.motion_20261119.reference.g1_pingpong.pd_gains.JOINT_PD:pingpong",
    },
  }

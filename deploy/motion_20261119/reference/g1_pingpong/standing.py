"""Frozen first-frame tracking with measured or timed readiness for sim and DDS.

The Stage 1 tracker closes the balance loop; Stage 2 and its motion clock stay
idle. Use MotionCommand's frame-zero freeze convention (last_dt=0). Sustained
holding must still be validated for each tracker/motion pair.
"""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.commands import CommandLimiter
from deploy.motion_20261119.reference.g1_pingpong.core import Controller, rotation


class StandingController:
  def __init__(self, cfg, motion, tracker, runtime):
    self.motion = motion
    self.controller = Controller(cfg, motion, tracker)
    self.controller.delta_dt = 0.0
    self.dt = cfg["control_dt"]
    self.limiter = CommandLimiter(
      cfg["robot"]["joint_limits"], runtime["guard"], self.dt
    )
    self.minimum_seconds = runtime["transition_seconds"]
    self.recovery_seconds = runtime.get("recovery_seconds", self.minimum_seconds)
    if not np.isfinite(self.recovery_seconds) or self.recovery_seconds < 0:
      raise ValueError("Recovery delay must be finite and nonnegative")
    self.readiness_mode = runtime.get("readiness_mode", "measured")
    if self.readiness_mode not in ("measured", "timed"):
      raise ValueError("readiness_mode must be measured or timed")
    self.options = {
      "dwell_seconds": 1.0,
      "timeout_seconds": 15.0,
      "max_joint_velocity": 0.1,
      "max_gyro": 0.1,
      "max_tilt_degrees": 10.0,
      "max_anchor_velocity": 0.05,
      "max_height_error": 0.1,
      **runtime.get("standing", {}),
    }
    if any(not np.isfinite(v) or v <= 0 for v in self.options.values()):
      raise ValueError("Standing thresholds must be finite and positive")
    if self.options["timeout_seconds"] <= self.minimum_seconds:
      raise ValueError("Standing timeout must exceed transition_seconds")
    self.previous = None
    self.stable_since = None
    self.phase = None
    self.debug = {}
    self.last_anchor_position = None
    self.last_observation_time = None

  def observe(self, state, anchor_velocity=0.0):
    metrics = {
      "pose_error": float(np.max(np.abs(state.q - self.motion.q[0]))),
      "joint_velocity": float(np.max(np.abs(state.dq))),
      "gyro": float(np.max(np.abs(state.gyro))),
      "anchor_velocity": anchor_velocity,
      "height_error": float(abs(state.anchor_pos[2] - self.motion.sample(0)[1][2])),
      "tilt_degrees": float(
        np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
      ),
    }
    # Static tracking offsets do not establish whether the robot is moving or
    # upright. Keep pose error diagnostic; an explicit custom pose limit can
    # still constrain it. Height and measured speed rule out a settled collapse.
    stable = all(
      v <= self.options[f"max_{k}"]
      for k, v in metrics.items()
      if f"max_{k}" in self.options
    )
    return stable, metrics

  def resume_from(self, previous):
    self.controller.start_from(previous)
    self.controller.delta_dt = 0.0
    self.phase = None

  def update_readiness(self, state, lifecycle, now, previous_target=None):
    if lifecycle.state not in ("PREPARE", "READY", "ARMED", "RECOVER"):
      return None
    if lifecycle.state != self.phase:
      self.stable_since = None
      if lifecycle.state in ("PREPARE", "RECOVER"):
        self.previous = (
          state.q.copy()
          if previous_target is None or lifecycle.state == "RECOVER"
          else previous_target.copy()
        )
      self.phase = lifecycle.state
    if self.previous is None:
      self.previous = state.q.copy()
    anchor_velocity = 0.0
    if self.last_observation_time is not None and now > self.last_observation_time:
      anchor_velocity = float(
        np.linalg.norm(state.anchor_pos - self.last_anchor_position)
        / (now - self.last_observation_time)
      )
    self.last_anchor_position = state.anchor_pos.copy()
    self.last_observation_time = now
    stable, metrics = self.observe(state, anchor_velocity)
    self.stable_since = (
      (now if self.stable_since is None else self.stable_since) if stable else None
    )
    if lifecycle.state in ("PREPARE", "RECOVER"):
      elapsed = now - lifecycle.transition_start
      required_seconds = (
        self.recovery_seconds if lifecycle.state == "RECOVER" else self.minimum_seconds
      )
      if elapsed >= required_seconds and (
        self.readiness_mode == "timed"
        or (
          self.stable_since is not None
          and now - self.stable_since >= self.options["dwell_seconds"]
        )
      ):
        lifecycle.ready()
      elif elapsed >= self.options["timeout_seconds"]:
        raise TimeoutError("First-frame standing did not stabilize before timeout")
    elif not stable and self.readiness_mode == "measured":
      # Withdraw READY/ARMED permission when measured stability is lost.
      lifecycle.restabilize(state.q, now)
    self.debug = {
      **metrics,
      "stable": stable,
      "readiness_mode": self.readiness_mode,
      "stable_seconds": (
        now - self.stable_since if self.stable_since is not None else 0.0
      ),
      "reference_time": 0.0,
    }

  def command(self, state):
    # Advance action history only when this controller supplies the command
    # for this tick. Selection can transfer control after the readiness check.
    observation = self.controller.observation(state)
    action = np.asarray(self.controller.tracker(observation))
    if action.shape != (29,) or not np.isfinite(action).all():
      raise ValueError("Invalid standing tracker action")
    desired = self.controller.default + self.controller.scale * action
    # Preparation conditions the initial handover. Once balancing, use the
    # configured policy command contract, also used during swings. Silently
    # reapplying disabled target limits here changes the PD support torques.
    target = self.limiter.apply(
      desired, state.q, self.previous, preparation=self.phase == "PREPARE"
    )
    # Training stores the policy's joint action, before actuator-side shaping.
    self.controller.previous_joint = action.astype(np.float32)
    self.controller.debug = {
      "reference_time": 0.0,
      "reference_delta": self.controller.delta_dt,
      "tracker_obs": observation,
      "base_action": action,
      "joint_action": action,
      "target_q": desired,
      "limited_target_q": target,
    }
    self.previous = target.copy()
    self.debug.update(
      limited=bool(np.any(np.abs(desired - target) > 1e-6)),
      raw_target_q=desired.tolist(),
    )
    return target

  def step(self, state, lifecycle, now, previous_target=None):
    if lifecycle.state not in ("PREPARE", "READY", "ARMED", "RECOVER"):
      return None
    self.update_readiness(state, lifecycle, now, previous_target)
    return self.command(state)

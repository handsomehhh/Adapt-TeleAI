"""Bounded virtual XY targets for a reference-tracking actor.

W: hope_table world. H: current pelvis yaw heading (X forward, Y left).
This is an observation shaper, not a velocity controller or reachability model.
The caller retains the physical goal; no input arrays are modified.
"""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.motions import heading


class BaseTargetCompensator:
  DEFAULTS = dict(
    mode="off", stages=["pre_strike"],
    position_gain=[0.3, 0.5], velocity_error_gain_s=[0.10, 0.10],
    desired_speed_limit_mps=[0.35, 0.70], time_reserve_s=0.08,
    minimum_time_s=0.15, max_offset_heading_m=[0.03, 0.08],
    offset_rate_limit_mps=[0.10, 0.20], velocity_filter_tau_s=0.08,
    arrival_tolerance_m=[0.02, 0.02], fade_distance_m=[0.06, 0.10],
    fade_out_time_s=0.10, target_jump_threshold_m=0.15,
    max_sample_gap_s=0.10, max_measured_speed_mps=3.0,
  )

  def __init__(self, config=None):
    if config is None:
      config = {}
    if not isinstance(config, dict) or set(config) - self.DEFAULTS.keys():
      raise ValueError("Invalid base_target_compensation mapping or unknown parameter")
    cfg = {**self.DEFAULTS, **config}
    if cfg["mode"] not in ("off", "shadow", "active"):
      raise ValueError('base_target_compensation.mode must be "off", shadow or active (quote "off" in YAML)')
    if cfg["stages"] != ["pre_strike"]:
      raise ValueError("base_target_compensation currently supports stages: [pre_strike] only")
    self.mode = cfg["mode"]
    for name in ("position_gain", "velocity_error_gain_s", "desired_speed_limit_mps",
                 "max_offset_heading_m", "offset_rate_limit_mps", "arrival_tolerance_m", "fade_distance_m"):
      value = np.asarray(cfg[name], dtype=float)
      if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"base_target_compensation.{name} must be two finite nonnegative values")
      if name in ("desired_speed_limit_mps", "offset_rate_limit_mps", "fade_distance_m") and np.any(value <= 0):
        raise ValueError(f"base_target_compensation.{name} must be positive")
      setattr(self, name, value)
    if np.any(self.fade_distance_m <= self.arrival_tolerance_m):
      raise ValueError("fade_distance_m must exceed arrival_tolerance_m on each axis")
    for name in ("time_reserve_s", "minimum_time_s", "velocity_filter_tau_s", "fade_out_time_s",
                 "target_jump_threshold_m", "max_sample_gap_s", "max_measured_speed_mps"):
      value = float(cfg[name])
      if not np.isfinite(value) or value < 0 or (name != "time_reserve_s" and value == 0):
        raise ValueError(f"Invalid base_target_compensation.{name}")
      setattr(self, name, value)
    self.reset()

  def reset(self):
    self.previous_position = self.previous_time = self.previous_target = None
    self.filtered_velocity_world = np.zeros(2)
    self.offset_world = np.zeros(2)
    self.draining = False

  def begin(self):
    """A new swing starts from zero bias; retain measured velocity history."""
    self.offset_world[:] = 0.
    self.previous_target = None
    self.draining = False

  def update(self, state, true_target_world_xy, *, now, time_to_strike, swinging):
    target = np.asarray(true_target_world_xy, dtype=float).copy()
    debug = dict(mode=self.mode, true_target_world_xy=target.copy(),
                 virtual_target_world_xy=target.copy(), actor_target_world_xy=target.copy(),
                 applied_offset_world_xy=np.zeros(2), proposed_offset_world_xy=np.zeros(2),
                 reason="disabled", stage="pre_strike" if swinging and time_to_strike > 0 else
                 ("follow_through" if swinging else "idle"))
    if self.mode == "off":
      return target, debug

    def fallback(reason):
      self.reset()
      debug["reason"] = reason
      return target, debug

    root_position = np.asarray(state.root_pos, dtype=float)
    if root_position.shape != (3,):
      return fallback("invalid_input")
    pos = root_position[:2]
    if (target.shape != (2,) or pos.shape != (2,)
        or not np.isfinite(np.r_[pos, target, now, time_to_strike]).all()):
      return fallback("invalid_input")
    try:
      yaw = heading(state.root_quat)
    except (ValueError, TypeError, IndexError):
      return fallback("invalid_heading")
    if not np.isfinite(yaw):
      return fallback("invalid_heading")
    c, s = np.cos(yaw), np.sin(yaw)
    h_to_w = np.array([[c, -s], [s, c]])
    dt = None if self.previous_time is None else now - self.previous_time
    if dt is not None and (dt <= 0 or dt > self.max_sample_gap_s):
      fallback("invalid_sample_interval")
      self.previous_position, self.previous_time = pos.copy(), now
      return target, debug
    measured = None
    if dt is not None:
      measured = (pos - self.previous_position) / dt
      if not np.isfinite(measured).all() or np.linalg.norm(measured) > self.max_measured_speed_mps:
        fallback("position_jump")
        self.previous_position, self.previous_time = pos.copy(), now
        return target, debug
    self.previous_position, self.previous_time = pos.copy(), now
    # Use the same supplied odometry as the actor when present, never plant velocity.
    if hasattr(state, "root_velocity_world"):
      supplied = np.asarray(state.root_velocity_world, dtype=float)
      if (supplied.shape != (3,) or not np.isfinite(supplied).all()
          or np.linalg.norm(supplied[:2]) > self.max_measured_speed_mps):
        return fallback("invalid_velocity")
      measured = supplied[:2]
    if dt is None:
      if measured is not None:
        self.filtered_velocity_world = measured.copy()
      self.previous_target = target.copy()
      debug["reason"] = "warming_up"
      return target, debug
    alpha = -np.expm1(-dt / self.velocity_filter_tau_s)
    self.filtered_velocity_world += alpha * (measured - self.filtered_velocity_world)
    error = h_to_w.T @ (target - pos)
    velocity = h_to_w.T @ self.filtered_velocity_world
    distance = np.abs(error)
    direction = np.sign(error)
    desired = np.minimum(distance / max(time_to_strike-self.time_reserve_s, self.minimum_time_s),
                         self.desired_speed_limit_mps)
    approaching = direction * velocity
    fade = np.clip((distance-self.arrival_tolerance_m) /
                   (self.fade_distance_m-self.arrival_tolerance_m), 0., 1.)
    fade = fade * fade * (3.-2.*fade)
    speed_gate = np.clip(1.-approaching/np.maximum(desired, 1e-6), 0., 1.)
    phase_gate = np.clip(time_to_strike/self.fade_out_time_s, 0., 1.) if swinging else 0.
    limit = self.max_offset_heading_m * fade * speed_gate * phase_gate
    requested = direction * np.clip(self.position_gain*distance +
                                    self.velocity_error_gain_s*(desired-approaching), 0., self.max_offset_heading_m)
    requested *= fade * speed_gate * phase_gate
    reason = "tracking"
    if self.previous_target is not None and np.linalg.norm(target-self.previous_target) > self.target_jump_threshold_m:
      self.draining = True
    self.previous_target = target.copy()
    if not swinging:
      # Home is a different physical goal. Never carry a swing bias into idle.
      self.offset_world[:] = 0.
      self.draining = False
      reason = "idle"
    elif self.draining:
      requested[:] = 0.
      reason = "target_jump_decay"
    elif time_to_strike <= 0:
      reason = "follow_through_decay"
    elif not np.any(fade):
      reason = "within_position_tolerance"
    elif not np.any(speed_gate):
      reason = "speed_sufficient"
    old_offset = h_to_w.T @ self.offset_world
    offset = old_offset + np.clip(requested-old_offset,
                                  -self.offset_rate_limit_mps*dt, self.offset_rate_limit_mps*dt)
    # Withdrawal to the current safe envelope takes precedence over slew limits:
    # no residual pushing past a goal, after impact, or in the wrong direction.
    projected = direction * np.clip(direction*offset, 0., limit)
    envelope_override = bool(np.any(np.abs(projected-offset) > 1e-10))
    self.offset_world = h_to_w @ projected
    if self.draining and np.linalg.norm(self.offset_world) < 1e-9:
      self.draining = False
    virtual = target + self.offset_world
    applied = self.offset_world if self.mode == "active" else np.zeros(2)
    debug.update(
      reason=reason, error_heading_xy=error, velocity_heading_xy=velocity,
      desired_velocity_heading_xy=direction*desired,
      proposed_offset_heading_xy=projected, proposed_offset_world_xy=self.offset_world.copy(),
      applied_offset_world_xy=applied.copy(), virtual_target_world_xy=virtual,
      actor_target_world_xy=target+applied, envelope_override=envelope_override,
      within_position_tolerance=bool(np.all(distance <= self.arrival_tolerance_m)),
    )
    return target+applied, debug

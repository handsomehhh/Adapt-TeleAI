"""Base XY target transitions between idle/home and swing goals.

The executor's physical base goal switches discretely: idle -> swing at motion
selection, swing -> idle at swing completion, and on the first tick after a
policy (re)entry. A transition mechanism decides what the actor actually sees
while the goal switches. All XY values are hope_table world (W) coordinates.

Mechanisms (``base_target_transition.mode``):

* ``step`` (default): the actor sees the new goal immediately (original behavior).
* ``smooth``: at each switch, latch the measured pelvis XY and move the actor
  target from there to the goal along a minimum-jerk (quintic) profile. The
  duration follows from peak speed/acceleration limits; a swing transition is
  additionally capped so it completes ``swing_arrival_reserve_s`` before the
  strike. Live goal updates during a transition are followed continuously:
  ``target = start + s(t) * (goal_now - start)``.

Only the actor's base target input changes. Motion selection, reachability
checks and the latched physical goal keep using the real goal.
"""

import numpy as np

SEGMENTS = ("entry", "swing", "return")
# Minimum-jerk s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: peak ds/dtau = 15/8 at
# tau = 1/2, peak |d2s/dtau2| = 10/sqrt(3) at tau = 1/2 -+ sqrt(3)/6.
MIN_JERK_PEAK_VELOCITY = 15. / 8.
MIN_JERK_PEAK_ACCELERATION = 10. / np.sqrt(3.)


def min_jerk(tau):
  tau = float(np.clip(tau, 0., 1.))
  return tau**3 * (10. + tau * (-15. + 6. * tau))


def segment_name(previous_swinging, swinging):
  if previous_swinging is None:
    return "swing" if swinging else "entry"
  return "swing" if swinging else "return"


class StepTransition:
  """Original behavior: the actor target equals the physical goal at once."""

  mode = "step"
  DEFAULTS = dict(mode="step")

  def __init__(self, config):
    self.config = dict(config)
    self.reset()

  def reset(self):
    self.previous_swinging = None

  def update(self, state, goal_xy, *, now, swinging, time_to_strike):
    goal = np.asarray(goal_xy, dtype=float).copy()
    segment = None
    if self.previous_swinging is None or self.previous_swinging != swinging:
      segment = segment_name(self.previous_swinging, swinging)
    self.previous_swinging = swinging
    return goal, dict(mode=self.mode, goal_world_xy=goal.copy(), target_world_xy=goal.copy(),
                      new_segment=segment, progress=1., reason="step")


class SmoothTransition:
  """Minimum-jerk interpolation from the measured base XY to the goal."""

  mode = "smooth"
  # Chosen from real 2026-11-16/17 logs plus MuJoCo step probes; see
  # experiments/base_target_transition_20261117/README.md. Short, nearly fixed
  # duration: the actor lags ~0.2 s, and the next swing may start 0.5 s after
  # a return, when residual base speed already doubles the miss rate. Swing
  # starts are small (median 3 cm), unrelated to tilt/misses: not smoothed.
  DEFAULTS = dict(
    mode="smooth", segments=["entry", "return"], max_speed_mps=1.0, max_accel_mps2=8.0,
    min_duration_s=0.2, max_duration_s=0.25, swing_arrival_reserve_s=0.1,
  )

  def __init__(self, config):
    cfg = {**self.DEFAULTS, **config}
    segments = cfg["segments"]
    if (not isinstance(segments, (list, tuple)) or len(set(segments)) != len(segments)
        or any(s not in SEGMENTS for s in segments)):
      raise ValueError(f"base_target_transition.segments must be a list drawn from {list(SEGMENTS)}")
    self.segments = tuple(segments)
    for name in ("max_speed_mps", "max_accel_mps2", "min_duration_s", "max_duration_s",
                 "swing_arrival_reserve_s"):
      value = cfg[name]
      if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"base_target_transition.{name} must be a finite number")
      value = float(value)
      if value < 0 or (name in ("max_speed_mps", "max_accel_mps2") and value == 0):
        raise ValueError(f"Invalid base_target_transition.{name}")
      setattr(self, name, value)
    if self.max_duration_s < self.min_duration_s:
      raise ValueError("base_target_transition.max_duration_s must be >= min_duration_s")
    self.reset()

  def reset(self):
    self.previous_swinging = None
    self.segment = None
    self.start_xy = self.start_time = None
    self.duration = 0.

  def duration_for(self, distance, segment, time_to_strike):
    """Shortest min-jerk time within speed/accel limits, clipped, deadline-capped."""
    kinematic = max(MIN_JERK_PEAK_VELOCITY * distance / self.max_speed_mps,
                    np.sqrt(MIN_JERK_PEAK_ACCELERATION * distance / self.max_accel_mps2))
    duration = float(np.clip(kinematic, self.min_duration_s, self.max_duration_s))
    if segment == "swing":
      # A base arriving after the strike misses the ball: the deadline wins
      # over the smoothness limits, down to a plain step.
      available = time_to_strike - self.swing_arrival_reserve_s
      duration = min(duration, available) if np.isfinite(available) else duration
    return max(duration, 0.)

  def update(self, state, goal_xy, *, now, swinging, time_to_strike):
    goal = np.asarray(goal_xy, dtype=float).copy()
    debug = dict(mode=self.mode, goal_world_xy=goal.copy(), target_world_xy=goal.copy(),
                 new_segment=None, segment=self.segment, progress=1., duration_s=self.duration,
                 start_world_xy=None, reason="arrived")
    if self.previous_swinging is None or self.previous_swinging != swinging:
      segment = segment_name(self.previous_swinging, swinging)
      self.previous_swinging = swinging
      debug["new_segment"] = segment
      self.segment, self.start_xy, self.start_time, self.duration = segment, None, None, 0.
      debug["segment"] = segment
      if segment not in self.segments:
        debug["reason"] = "segment_not_smoothed"
        return goal, debug
      start = np.asarray(state.root_pos, dtype=float)[:2].copy()
      if start.shape != (2,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
        debug["reason"] = "invalid_input"
        return goal, debug
      distance = float(np.linalg.norm(goal - start))
      self.start_xy, self.start_time = start, float(now)
      self.duration = self.duration_for(distance, segment, float(time_to_strike))
      debug.update(duration_s=self.duration, start_distance_m=distance)
    if self.start_xy is None:
      debug["reason"] = "segment_not_smoothed" if self.segment not in self.segments else debug["reason"]
      return goal, debug
    elapsed = max(float(now) - self.start_time, 0.)
    progress = 1. if self.duration <= 0 else min(elapsed / self.duration, 1.)
    s = min_jerk(progress)
    target = self.start_xy + s * (goal - self.start_xy)
    debug.update(target_world_xy=target.copy(), start_world_xy=self.start_xy.copy(),
                 progress=progress, elapsed_s=elapsed, duration_s=self.duration,
                 reason="arrived" if progress >= 1. else "transitioning")
    if progress >= 1.:
      # Hold the goal itself afterwards so later live updates apply directly.
      self.start_xy = None
    return target, debug


TRANSITIONS = {cls.mode: cls for cls in (StepTransition, SmoothTransition)}


def build_base_target_transition(config=None):
  """``config``: None, or a mapping with ``mode`` (step/smooth) and its parameters."""
  config = {} if config is None else config
  if not isinstance(config, dict):
    raise ValueError("base_target_transition must be a mapping")
  mode = config.get("mode", "step")
  if mode not in TRANSITIONS:
    raise ValueError(f"base_target_transition.mode must be one of {sorted(TRANSITIONS)}")
  # Parameters of every mechanism may stay in the YAML, so switching mode
  # (e.g. from the command line) needs no other edit; only typos are rejected.
  known = set().union(*(cls.DEFAULTS for cls in TRANSITIONS.values()))
  unknown = set(config) - known
  if unknown:
    raise ValueError(f"Unknown base_target_transition parameters: {sorted(unknown)}")
  cls = TRANSITIONS[mode]
  return cls({k: v for k, v in config.items() if k in cls.DEFAULTS})

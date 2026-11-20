"""Timestamped HOPE ball-position input, velocity fitting and receive planning."""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import BallState, quaternion_multiply, rotation


@dataclass
class BallEstimate:
  position: np.ndarray
  velocity: np.ndarray
  stamp: float
  received: float


class BallEstimator:
  """Fit position/velocity at the newest source time, never callback time."""

  def __init__(
    self,
    window=9,
    max_gap=0.1,
    stale_timeout=0.1,
    minimum_samples=3,
    window_seconds=None,
    bounce_height=None,
    bounce_tolerance=0.03,
    bounce_min_speed=0.0,
  ):
    if window < 3 or max_gap <= 0 or stale_timeout <= 0:
      raise ValueError("Invalid estimator configuration")
    self.samples = deque(maxlen=window)
    # Preserve source-rate position packets for planners whose estimator must
    # see every mocap frame (the deployment BallEstimate remains unchanged).
    self._observation_queue = deque(maxlen=4096)
    self._observation_drops = 0
    self.max_gap, self.stale_timeout = max_gap, stale_timeout
    self.lock = threading.Lock()
    self.latest = None
    if not 3 <= minimum_samples <= window or (
      window_seconds is not None and window_seconds <= 0
    ):
      raise ValueError("Invalid polynomial fit window/minimum_samples")
    self.minimum_samples, self.window_seconds = minimum_samples, window_seconds
    self.bounce_height, self.bounce_tolerance = bounce_height, bounce_tolerance
    if bounce_min_speed < 0 or not np.isfinite(bounce_min_speed):
      raise ValueError("Invalid minimum bounce speed")
    self.bounce_min_speed = bounce_min_speed

  def update(self, position, stamp, *, received=None):
    received = time.monotonic() if received is None else received
    position = np.asarray(position, dtype=float)
    if (
      position.shape != (3,)
      or not np.isfinite(position).all()
      or not np.isfinite(stamp)
    ):
      raise ValueError("Invalid ball sample")
    with self.lock:
      if len(self._observation_queue) == self._observation_queue.maxlen:
        self._observation_drops += 1
      self._observation_queue.append((float(stamp), position.copy(), received))
      if self.samples and stamp <= self.samples[-1][0]:
        return False
      if self.samples and stamp - self.samples[-1][0] > self.max_gap:
        self.samples.clear()
        self.latest = None
      # A3 Stage 1 clears the polynomial across a table-bounce discontinuity.
      # Account for the sphere radius and finite sampling rate in the height
      # tolerance; do not fit descending and ascending arcs as one parabola.
      if self.bounce_height is not None and len(self.samples) >= 2:
        z0, z1, z2 = self.samples[-2][1][2], self.samples[-1][1][2], position[2]
        t0, t1 = self.samples[-2][0], self.samples[-1][0]
        if (
          z0 > z1 < z2
          and abs(z1 - self.bounce_height) <= self.bounce_tolerance
          and (z0 - z1) / (t1 - t0) >= self.bounce_min_speed
          and (z2 - z1) / (stamp - t1) >= self.bounce_min_speed
        ):
          self.samples.clear()
          self.latest = None
      self.samples.append((stamp, position.copy()))
      if self.window_seconds is not None:
        while self.samples and stamp - self.samples[0][0] > self.window_seconds:
          self.samples.popleft()
      if len(self.samples) >= self.minimum_samples:
        times = np.array([s[0] - stamp for s in self.samples])
        design = np.column_stack([np.ones(len(times)), times, times**2])
        coefficients = np.linalg.lstsq(
          design, np.stack([s[1] for s in self.samples]), rcond=None
        )[0]
        self.latest = BallEstimate(coefficients[0], coefficients[1], stamp, received)
    return True

  def drain_observations(self):
    """Return and remove raw ``(source_stamp, position, received)`` packets."""
    with self.lock:
      rows = list(self._observation_queue)
      self._observation_queue.clear()
    return rows

  def get(self, now=None):
    now = time.monotonic() if now is None else now
    with self.lock:
      if self.latest is None or now - self.latest.received > self.stale_timeout:
        return None
      item = self.latest
      return BallEstimate(
        item.position.copy(), item.velocity.copy(), item.stamp, item.received
      )


class RootPoseCache:
  def __init__(
    self,
    offset,
    stale_timeout=0.1,
    orientation_offset=(1.0, 0.0, 0.0, 0.0),
    conditioning=None,
  ):
    self.offset = np.asarray(offset, dtype=float)
    self.stale_timeout = stale_timeout
    self.orientation_offset = np.asarray(orientation_offset)
    rotation(self.orientation_offset)
    self.latest = None
    self.lock = threading.Lock()
    from deploy.motion_20261119.reference.io.pose_conditioning import PoseConditioner

    self.conditioner = PoseConditioner(conditioning) if conditioning else None
    self.velocity_history = deque(maxlen=16)

  def update(self, position, quaternion, stamp, *, received=None):
    received = time.monotonic() if received is None else received
    quaternion = np.asarray(quaternion, dtype=float)
    r = rotation(quaternion)
    position = np.asarray(position, dtype=float)
    if (
      position.shape != (3,)
      or not np.isfinite(position).all()
      or not np.isfinite(stamp)
    ):
      raise ValueError("Invalid root pose")
    with self.lock:
      if self.latest is not None and stamp <= self.latest[2]:
        return
      quaternion = quaternion_multiply(quaternion, self.orientation_offset)
      position = position + r @ self.offset
      if self.conditioner is not None:
        conditioned = self.conditioner.update(position, quaternion, stamp, received)
        if conditioned is None:
          return
        position, quaternion = conditioned
        self.velocity_history.append((stamp, self.conditioner.velocity.copy()))
      self.latest = (
        position,
        quaternion / np.linalg.norm(quaternion),
        stamp,
        received,
      )

  def get(self, now=None):
    now = time.monotonic() if now is None else now
    with self.lock:
      if self.latest is None or now - self.latest[3] > self.stale_timeout:
        return None
      return tuple(v.copy() if isinstance(v, np.ndarray) else v for v in self.latest)

  def velocity_for_source(self, stamp):
    """Return velocity from the same accepted pose, even if a callback advanced."""
    with self.lock:
      for source, velocity in reversed(self.velocity_history):
        if source == stamp:
          return velocity.copy()
    return None


def predict(position, velocity, duration, gravity=9.81, drag=0.0):
  """Match training's second-order, quadratic-drag prediction observation."""
  acceleration = (
    np.array([0.0, 0.0, -gravity]) - drag * np.linalg.norm(velocity) * velocity
  )
  return position + velocity * duration + 0.5 * acceleration * duration**2


def intercept_without_drag(position, velocity, plane_x, cfg, horizon):
  """Exact flight events with the same finite-table bounce as the simulator."""
  p, v, elapsed = position.copy(), velocity.copy(), 0.0
  gravity, height = cfg["gravity"], cfg["table_height"] + cfg["ball_radius"]
  for _ in range(4):
    if v[0] >= -0.1 or p[0] < plane_x:
      return None
    crossing = (plane_x - p[0]) / v[0]
    if elapsed + crossing > horizon:
      return None
    bounce_time = np.inf
    discriminant = v[2] ** 2 + 2 * gravity * (p[2] - height)
    # A ball below the tabletop can rise above it before the later descending
    # crossing. Choose that future root, rather than dropping the bounce.
    if discriminant >= 0:
      candidate = (v[2] + np.sqrt(discriminant)) / gravity
      contact = predict(p, v, candidate, gravity)
      if (
        candidate >= 0
        and cfg["table_near_x"] <= contact[0] <= cfg["table_far_x"]
        and abs(contact[1] - cfg.get("table_center_y", 0.0)) <= cfg["table_width"] / 2
      ):
        bounce_time = candidate
    if crossing <= bounce_time:
      return predict(p, v, crossing, gravity), elapsed + crossing
    p = predict(p, v, bounce_time, gravity)
    p[2] = height
    v[2] -= gravity * bounce_time
    v[:2] *= 0.96
    v[2] *= -cfg["table_restitution"]
    elapsed += bounce_time
  return None


class ReceivePlanner:
  """Predict a plane intercept and latch a timely, reachable receive plan.

  RallyController scores motions against this forecast. Once triggered, the
  intercept/deadline stay fixed while measured ball state continues updating.
  """

  def __init__(
    self,
    motion,
    ball_cfg,
    spatial_tolerance=(0.12, 0.25, 0.08),
    timing_tolerance=0.18,
    max_start_frame=None,
  ):
    self.cfg = ball_cfg
    self.strike = motion.strike_position(ball_cfg)
    self.strike_time = motion.strike_time(ball_cfg)
    self.target = np.array(
      [*ball_cfg["target_xy"], ball_cfg["table_height"] + ball_cfg["ball_radius"]]
    )
    self.spatial_tolerance = np.asarray(spatial_tolerance)
    self.timing_tolerance = timing_tolerance
    self.max_start_frame = max_start_frame
    self.fps = motion.fps
    self.reset()

  def reset(self):
    self.plan = None
    self.hit = False
    self.previous_velocity = None
    self.diagnostic = {"reason": "waiting_for_incoming_ball"}

  def try_start(self, estimate, now):
    age = max(now - estimate.received, 0)
    p = predict(
      estimate.position, estimate.velocity, age, self.cfg["gravity"], self.cfg["drag"]
    )
    acceleration = np.array([0.0, 0.0, -self.cfg["gravity"]])
    acceleration -= (
      self.cfg["drag"] * np.linalg.norm(estimate.velocity) * estimate.velocity
    )
    v = estimate.velocity + acceleration * age
    if v[0] >= -0.1:
      self.diagnostic = {"reason": "not_incoming", "vx": float(v[0])}
      return False
    if self.cfg["drag"] == 0:
      prediction = intercept_without_drag(
        p, v, self.strike[0], self.cfg, self.strike_time + self.timing_tolerance
      )
      if prediction is None:
        self.diagnostic = {"reason": "no_strike_plane_crossing_in_start_window"}
        return False
      intercept, flight = prediction
      return self.accept(intercept, flight, now)
    # Integrate the same quadratic-drag flight model until the clip strike plane.
    flight, previous = 0.0, p.copy()
    while flight < self.strike_time + self.timing_tolerance:
      next_p = predict(p, v, 0.001, self.cfg["gravity"], self.cfg["drag"])
      a = (
        np.array([0.0, 0.0, -self.cfg["gravity"]])
        - self.cfg["drag"] * np.linalg.norm(v) * v
      )
      v = v + a * 0.001
      height = self.cfg["table_height"] + self.cfg["ball_radius"]
      if p[2] >= height >= next_p[2] and v[2] < 0:
        fraction = (p[2] - height) / max(p[2] - next_p[2], 1e-8)
        contact = p + fraction * (next_p - p)
        if (
          self.cfg["table_near_x"] <= contact[0] <= self.cfg["table_far_x"]
          and abs(contact[1]) <= self.cfg["table_width"] / 2
        ):
          v[:2] *= 0.96
          v[2] *= -self.cfg["table_restitution"]
          next_p = contact + v * ((1 - fraction) * 0.001)
      flight += 0.001
      if previous[0] >= self.strike[0] >= next_p[0]:
        fraction = (previous[0] - self.strike[0]) / (previous[0] - next_p[0])
        intercept = previous + fraction * (next_p - previous)
        flight -= (1 - fraction) * 0.001
        return self.accept(intercept, flight, now)
      previous, p = next_p, next_p
    self.diagnostic = {"reason": "no_strike_plane_crossing_in_start_window"}
    return False

  def accept(self, intercept, flight, now):
    self.diagnostic = {
      "remaining": float(flight),
      "intercept": intercept.tolist(),
      "strike_time": float(self.strike_time),
      "position_error": (intercept - self.strike).tolist(),
    }
    # HOPE triggers when TTS falls to the trained pre-strike horizon. Tolerance
    # permits a late sample, not starting up to 80 ms before that horizon.
    late_window = (
      self.timing_tolerance
      if self.max_start_frame is None
      else self.max_start_frame / self.fps
    )
    if flight > self.strike_time or flight < max(0.05, self.strike_time - late_window):
      self.diagnostic["reason"] = "arrival_outside_start_window"
      return False
    self.diagnostic["start_frame"] = max(
      0, round((self.strike_time - flight) * self.fps)
    )
    if np.any(np.abs(intercept - self.strike) > self.spatial_tolerance):
      self.diagnostic["reason"] = "intercept_outside_motion_envelope"
      return False
    self.diagnostic["reason"] = "accepted"
    self.plan = (intercept, now + flight)
    return True

  def observation(self, estimate, now, racket_pos):
    if self.plan is None:
      raise RuntimeError("No latched receive plan")
    age = max(now - estimate.received, 0)
    p = predict(
      estimate.position, estimate.velocity, age, self.cfg["gravity"], self.cfg["drag"]
    )
    a = (
      np.array([0.0, 0.0, -self.cfg["gravity"]])
      - self.cfg["drag"] * np.linalg.norm(estimate.velocity) * estimate.velocity
    )
    v = estimate.velocity + a * age
    if self.previous_velocity is not None:
      # Approximate measured contact, rather than setting hit unconditionally.
      nearby = np.linalg.norm(p - racket_pos) < self.cfg["racket_radius"] + 0.05
      self.hit |= bool(nearby and self.previous_velocity[0] < 0 and v[0] > 0)
    self.previous_velocity = v.copy()
    intercept, deadline = self.plan
    return BallState(p, v, intercept, deadline - now, self.target, self.hit)


def sample_received_time(stamp, config, *, wall_time=None, monotonic_time=None):
  """Translate a validated source wall-clock stamp to local monotonic time."""
  wall_time = time.time() if wall_time is None else wall_time
  monotonic_time = time.monotonic() if monotonic_time is None else monotonic_time
  age = wall_time - (stamp + config["source_clock_offset"])
  if not np.isfinite(age) or not (-0.02 <= age <= config["source_max_age"]):
    raise ValueError("Mocap sample source time is expired or clocks differ")
  return monotonic_time - max(age, 0)


class BallEpisodeMonitor:
  """Observed counterpart of training's landed/missed episode boundary.

  Uses measured trajectories only, never simulator hit/landing flags or IDs.
  A new discontinuous incoming track ends the old residual-policy context.
  """

  def __init__(self, cfg):
    self.cfg = cfg
    self.previous = None
    self.outgoing = False
    self.reason = None

  def update(self, ball, now):
    if self.reason is not None:
      return self.reason
    if ball is None:
      self.reason = "ball_observation_expired"
      return self.reason
    p, v = np.asarray(ball.position), np.asarray(ball.velocity)
    cfg = self.cfg
    if (
      p[2] < cfg["ball_radius"]
      or p[0] < -1
      or p[0] > cfg["table_far_x"] + 2
      or abs(p[1]) > 2.5
    ):
      self.reason = "observed_out_of_play"
    elif self.previous is not None:
      previous_p, previous_v, previous_time = self.previous
      dt = now - previous_time
      if (
        0 < dt <= 0.1
        and np.linalg.norm(
          p - predict(previous_p, previous_v, dt, cfg["gravity"], cfg["drag"])
        )
        > 0.5
      ):
        self.reason = "new_ball_track"
      elif (
        self.outgoing
        and previous_v[2] < -0.1
        and v[2] > 0.1
        and cfg["table_near_x"] <= p[0] <= cfg["table_far_x"]
        and abs(p[1]) <= cfg["table_width"] / 2
        and p[2] <= cfg["table_height"] + cfg["ball_radius"] + 0.25
      ):
        self.reason = "observed_outgoing_landing"
    self.outgoing |= bool(ball.hit or v[0] > 0.1)
    self.previous = (p.copy(), v.copy(), now)
    return self.reason

"""Cache host-estimated ball states without running a second trajectory fit."""

import threading
import time

import numpy as np

from deploy.g1_pingpong.perception import BallEstimate, sample_received_time


class SharedBallCache:
  """BallEstimator's read interface for a single shared host estimator.

  The sample timestamp, rather than UDP arrival, determines freshness. Clearing
  this cache never changes a controller's accepted shot or lifecycle state.
  """

  def __init__(self, config):
    self.config = config
    self.stale_timeout = float(config["estimator"]["stale_timeout"])
    if not np.isfinite(self.stale_timeout) or self.stale_timeout <= 0:
      raise ValueError("Invalid shared ball stale timeout")
    self.lock = threading.Lock()
    self.latest = None
    self.last_stamp = None
    self.last_emitted = None
    self.trajectory_epoch = None
    self.receiving_side = None

  def accept(self, message, *, wall_time, monotonic_time):
    if message.get("schema") != "g1_duel_ball_v1":
      raise ValueError("Invalid shared ball schema")
    valid = message.get("valid")
    epoch = message.get("trajectory_epoch")
    side = message.get("receiving_side")
    if (
      not isinstance(valid, bool)
      or type(epoch) is not int
      or epoch < 0
      or side not in (None, "P1", "P2")
    ):
      raise ValueError("Invalid shared ball validity, epoch or receiving side")
    emitted = float(message["emitted_monotonic"])
    if (
      not np.isfinite(emitted)
      or not -0.02 <= monotonic_time - emitted <= self.config["source_max_age"]
    ):
      raise ValueError("Shared ball emission time is expired or clocks differ")
    estimate = None
    if valid:
      stamp = float(message["stamp"])
      position = np.asarray(message["position"], dtype=float)
      velocity = np.asarray(message["velocity"], dtype=float)
      if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, velocity)):
        raise ValueError("Invalid shared ball position or velocity")
      received = sample_received_time(
        stamp, self.config, wall_time=wall_time, monotonic_time=monotonic_time
      )
      estimate = BallEstimate(position.copy(), velocity.copy(), stamp, received)
    with self.lock:
      if self.last_emitted is not None and emitted <= self.last_emitted:
        return False
      self.last_emitted = emitted
      if estimate is not None and self.last_stamp is not None:
        if estimate.stamp <= self.last_stamp:
          return False
      self.latest = estimate
      if estimate is not None:
        self.last_stamp = estimate.stamp
      self.trajectory_epoch = epoch
      self.receiving_side = side
    return True

  def get(self, now=None):
    now = time.monotonic() if now is None else now
    with self.lock:
      item = self.latest
      if item is None or now - item.received > self.stale_timeout:
        return None
      return BallEstimate(
        item.position.copy(), item.velocity.copy(), item.stamp, item.received
      )

  def drain_observations(self):
    """No raw positions are delivered to an embedded per-robot estimator."""
    return []

  def diagnostics(self):
    with self.lock:
      return {
        "trajectory_epoch": self.trajectory_epoch,
        "receiving_side": self.receiving_side,
      }

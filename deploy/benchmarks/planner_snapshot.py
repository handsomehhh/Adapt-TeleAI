"""Copyable, virtual-clock wrapper around the production HOPE planner core.

Real deployment keeps its worker/ROS transport. Independent benchmark resets
snapshot numerical state, never thread locks or transport queues. Processing is
synchronous at the delivered observation time, so browser speed cannot alter
which observations are available to a virtual control tick.
"""

import copy

import numpy as np

from deploy.racket_command import RacketCommand


class PlannerSnapshot:
  def __init__(self, source):
    self.planner = copy.deepcopy(source.planner)
    self.params = copy.deepcopy(source.params)
    self.latest = copy.deepcopy(source.latest)
    self.last_update_diagnostics = copy.deepcopy(source.last_update_diagnostics)
    self.origin = np.zeros(3)  # The current core already uses HOPE table XYZ.

  @property
  def config(self):
    return self.planner.config

  @property
  def diagnostic(self):
    return (
      self.latest.diagnostic
      if self.latest is not None
      else self.last_update_diagnostics
    )

  def update_observations(self, observations):
    for stamp, position, received in observations:
      command = self.planner.update(float(stamp), np.asarray(position, dtype=float))
      self.last_update_diagnostics = dict(self.planner.last_update_diagnostics)
      if command is None:
        continue
      remaining = float(command.t_strike) - float(stamp)
      self.latest = RacketCommand(
        position=np.asarray(command.p_intercept, dtype=float).copy(),
        velocity=np.asarray(command.v_racket, dtype=float).copy(),
        normal=np.asarray(command.n_racket, dtype=float).copy(),
        deadline=float(received) + remaining,
        diagnostic=dict(
          source="hope_planner_optimized_2_A30808",
          source_stamp=float(stamp),
          source_ball_position=np.asarray(position).tolist(),
          received_monotonic=float(received),
          time_to_strike=remaining,
          hit_plane_x=self.config.x_hit,
          position=np.asarray(command.p_intercept).tolist(),
          velocity=np.asarray(command.v_racket).tolist(),
          normal=np.asarray(command.n_racket).tolist(),
          outgoing_velocity=np.asarray(command.v_ball_outgoing).tolist(),
          target_land=np.asarray(command.target_land).tolist(),
          clears_net=bool(command.clears_net),
          bypasses_net_posts=bool(command.bypasses_net_posts),
          valid=bool(command.valid),
          num_bounces=int(command.num_bounces),
          stage1_fit_rmse_m=float(command.stage1_fit_rmse_m),
        ),
      )

  def latest_command(self, now):
    if self.latest is None or self.latest.deadline <= now:
      return None
    return self.latest

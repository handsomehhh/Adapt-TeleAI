"""Ball estimates -> original A3 planner/selector/actor; no direct commands."""

import copy

import numpy as np

from deploy.hope_g1.command_adapter import HopeCommandAdapter
from deploy.hope_g1.controller import HopeController


class HopeReceiveAdapter(HopeCommandAdapter):
  def __init__(self, *args, plane_x=-0.25, return_target=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.baseline.a3_options["hit_plane_x"] = plane_x
    planner_config = self.baseline.command_planner.config
    planner_config.x_hit = plane_x
    planner_config.publish_intercept_x_min = plane_x
    planner_config.publish_intercept_x_max = plane_x
    if return_target is not None:
      # A task goal input, shared with A3; planning arithmetic is unchanged.
      planner = self.baseline.command_planner
      planner.config.target_land = (
        np.r_[return_target, self.options["ball"]["table_height"]] - planner.origin
      )
    self.metadata.update(
      planner="Current production A30808 HOPEPlanner core; virtual-clock snapshot",
      command_source="Measured ball positions -> production Stage 1/2/3 -> command",
      direct_racket_commands=False,
      intercept_plane_x=plane_x,
      planner_return_target=(
        self.baseline.command_planner.config.target_land
        + self.baseline.command_planner.origin
      ).tolist(),
    )
    self.reset()

  def reset(self):
    super().reset()
    self.latched_command = None
    self.planner_records = []

  def observe_ball(self, position, stamp, received):
    self.engine.update_ball_observations(
      [(stamp, np.asarray(position).copy(), self.start_time + received)]
    )

  def advance_ball(self, estimate, on_physics_step=None):
    now = self.start_time + self.ticks * self.dt
    local = copy.deepcopy(estimate)
    if local is not None:
      local.received += self.start_time
      local.stamp += self.start_time
    # Preparation intentionally suppresses ball input. Invoke the original
    # receiving implementation here rather than that preparation-only shim.
    target = HopeController.step(self.engine, self.state(), local, now)
    self.decision = copy.deepcopy(self.engine.debug["decision"])
    diagnostic = copy.deepcopy(self.engine.command_planner.diagnostic)
    if self.engine.lifecycle.state == "ARMED" and local is not None:
      self.planner_records.append(
        dict(
          time=self.ticks * self.dt,
          estimate=dict(
            position=local.position.tolist(), velocity=local.velocity.tolist()
          ),
          planner=diagnostic,
          decision=self.decision,
        )
      )
    if not self.accepted and self.engine.lifecycle.state == "EXECUTE":
      self.accepted = True
      command = self.engine.raw_command
      self.latched_command = dict(
        position=command.position.tolist(),
        normal=command.normal.tolist(),
        velocity=command.velocity.tolist(),
        deadline=command.deadline - self.start_time,
        issued_at=self.ticks * self.dt,
        tts_at_issue=command.deadline - now,
        motion_index=self.engine.motion_index,
        diagnostic=copy.deepcopy(command.diagnostic),
        decision=self.decision,
      )
      self.planner_records.append(
        dict(
          time=self.ticks * self.dt,
          planner=diagnostic,
          decision=self.decision,
          latched=True,
        )
      )
      # One incoming ball per independent trial; do not swing at floor rebounds.
      self.engine.receiving = False
    self.integrate(target, on_physics_step)

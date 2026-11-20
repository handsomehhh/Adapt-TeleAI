"""A3 Stages 2/3 and RacketCommand semantics adapted to the G1 court frame.

Planning uses measured ball estimates only. It never reads the simulated ball
plant, writes ball velocity, or replaces joint actions emitted by the actor.
"""

from dataclasses import dataclass

import numpy as np

from deploy.motion_20261119.reference.hope_g1.a3_reference.ball_trajectory_predictor import (
  BallTrajectoryPredictor,
)
from deploy.motion_20261119.reference.hope_g1.a3_reference.constants import (
  BallPhysics,
  PlannerConfig,
  TableParams,
)
from deploy.motion_20261119.reference.hope_g1.a3_reference.racket_target_planner import RacketTargetPlanner
from deploy.motion_20261119.reference.racket_command import RacketCommand


@dataclass
class ReceivePlan:
  command: RacketCommand
  diagnostic: dict

  @property
  def plan(self):
    return self.command.position, self.command.deadline


class A3Planner:
  def __init__(self, ball, options):
    self.ball, self.options = ball, options
    # World and A3 normally coincide (near-left tabletop origin). Keep the
    # origin explicit so geometry contract tests can also exercise translations.
    self.origin = np.array(
      [
        ball["table_near_x"],
        ball.get("table_center_y", 0.0) + ball["table_width"] / 2,
        ball["table_height"],
      ]
    )
    self.physics = BallPhysics(
      k=ball["drag"],
      C_h=ball.get("table_tangential_damping", 0.75),
      C_v=ball["table_restitution"],
      g=np.array([0.0, 0.0, -ball["gravity"]]),
      radius=ball["ball_radius"],
    )
    self.table = TableParams(
      length=ball["table_far_x"] - ball["table_near_x"],
      width=ball["table_width"],
      height=ball["table_height"],
      net_x=(ball["table_far_x"] - ball["table_near_x"]) / 2,
      net_height=ball["net_height"],
      net_overhang=ball.get("net_overhang", TableParams().net_overhang),
    )
    self.config = PlannerConfig(
      dt_integrate=0.001,
      max_predict_time=2.0,
      target_land=np.r_[
        ball["target_xy"],
        ball.get("target_land_z", ball["table_height"]),
      ]
      - self.origin,
      delta_t_flight=float(options.get("flight_seconds", 0.5)),
      C_r=ball["racket_restitution"],
      racket_radius=ball["racket_radius"],
      time_to_strike_invalid_x=options.get("time_to_strike_invalid_x", 2.5),
    )
    if not 0 < self.config.delta_t_flight <= self.config.max_predict_time:
      raise ValueError("A3 flight_seconds must be in (0, 2]")
    self.predictor = BallTrajectoryPredictor(self.physics, self.config, self.table)
    self.target_planner = RacketTargetPlanner(self.physics, self.config, self.table)
    self.diagnostic = {}

  def plan(self, estimate, now, hit_plane_x):
    age = max(0.0, now - estimate.received)
    if age > self.options.get("command_timeout_seconds", 0.2):
      self.diagnostic = dict(reason="stale_ball_estimate", age=age)
      return None
    p, v = (
      np.asarray(estimate.position).copy() - self.origin,
      np.asarray(estimate.velocity).copy(),
    )
    if not np.isfinite(np.r_[p, v]).all() or v[0] >= 0:
      self.diagnostic = dict(reason="not_incoming")
      return None
    if p[0] > self.config.time_to_strike_invalid_x:
      self.diagnostic = dict(reason="ball_before_planning_window")
      return None
    # Start at the measurement epoch, preserving any bounce during its age.
    # Receiver-side TTS compensation is deadline-now, not a fresh timestamp.
    self.config.x_hit = hit_plane_x - self.origin[0]
    strike = self.predictor.predict(p, v, now - age)
    if not strike.valid:
      self.diagnostic = dict(reason="no_hit_plane_crossing")
      return None
    planned = self.target_planner.plan(strike)
    if (
      not planned.valid
      or not np.isfinite(np.r_[planned.v_racket, planned.n_racket]).all()
    ):
      self.diagnostic = dict(reason="invalid_a3_racket_command")
      return None
    diagnostic = dict(
      reason="planned",
      source="a3_optimized",
      estimate_stamp=float(estimate.stamp),
      age_seconds=age,
      remaining=float(planned.t_strike - now),
      hit_plane_x=float(hit_plane_x),
      predicted_bounces=int(strike.num_bounces),
      incoming_velocity=strike.v_ball.tolist(),
      position=(planned.p_intercept + self.origin).tolist(),
      velocity=planned.v_racket.tolist(),
      striking_normal=planned.n_racket.tolist(),
      predicted_outgoing_velocity=planned.v_ball_outgoing.tolist(),
      target_land=(planned.target_land + self.origin).tolist(),
      planned_net_clear=bool(planned.clears_net),
      bypasses_posts=bool(planned.bypasses_net_posts),
    )
    self.diagnostic = diagnostic
    return RacketCommand(
      planned.p_intercept + self.origin,
      planned.v_racket,
      planned.n_racket,
      float(planned.t_strike),
      diagnostic,
    )


def command_goal(command, reference_goal, reference, racket, hit_sign, mode, scale):
  """Port of A3 LatchRacketTargetFromCommand, keeping goals in deploy world.

  The actor builder performs the single final inverse rotation for absolute
  vectors. Raw mode flips the striking-face normal for backhand, never velocity.
  """
  position, velocity, normal = (
    command.position.copy(),
    command.velocity.copy(),
    command.normal.copy(),
  )
  if mode == "raw":
    normal = hit_sign * normal / np.linalg.norm(normal)
  elif mode == "envelope":
    inverse = reference.rotation.T
    offset = inverse @ (position - reference_goal.position)
    position = reference_goal.position + reference.rotation @ np.clip(
      offset,
      -np.asarray(racket["ref_perturb_pos"]) * scale,
      np.asarray(racket["ref_perturb_pos"]) * scale,
    )
    ref_normal = inverse @ reference_goal.normal
    ref_normal /= np.linalg.norm(ref_normal)
    requested = inverse @ normal / np.linalg.norm(normal)
    half = racket["ref_perturb_normal"] * scale
    mapped = ref_normal + np.clip(requested - ref_normal, -half, half)
    normal = reference.rotation @ (mapped / np.linalg.norm(mapped))
    half_speed = racket["ref_perturb_vel_scale"] * scale
    ratio = np.clip(
      np.linalg.norm(velocity) / racket["ref_velocity_speed"],
      1 - half_speed,
      1 + half_speed,
    )
    velocity = reference_goal.velocity * ratio
  else:
    raise ValueError("A3 target_mode must be raw or envelope")
  base_xy = reference_goal.base_xy + (position - reference_goal.position)[:2]
  return position, velocity, normal, base_xy

"""Continuous 104-D G1 hitter controller, ported from hope_mjlab play_1v0.

World: hope_table (tabletop Z=0). Training world has the same axes; its origin
is the robot station on the floor, so every target crosses only a translation.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from deploy.g1_pingpong.core import rotation
from deploy.g1_pingpong.lifecycle import Lifecycle
from deploy.hope_g1.controller import HopeStanding
from deploy.racket_command import decode_ros_planner_command

SIDES = ("forehand", "backhand")


@dataclass
class Shot:
  side: int
  position: np.ndarray
  velocity: np.ndarray
  strike_at: float
  station: np.ndarray
  diagnostic: dict


def observation(state, last_action, default_q, station, position, velocity, tts):
  """g1_hitter_pure_v1: world-axis differences, no yaw/base rotation."""
  root_rot = rotation(state.root_quat)
  forward = root_rot[:2, 0] / max(np.linalg.norm(root_rot[:2, 0]), 1e-6)
  values = np.concatenate(
    (
      state.gyro,
      state.q - default_q,
      state.dq,
      last_action,
      root_rot.T @ [0.0, 0.0, -1.0],
      forward,
      np.asarray(station) - state.root_pos[:2],
      np.asarray(position) - state.root_pos,
      velocity,
      [tts],
    )
  ).astype(np.float32)
  if values.shape != (104,) or not np.isfinite(values).all():
    raise ValueError("Invalid 104-D hitter observation")
  return values


class HitterActor:
  def __init__(self, cfg, policy):
    robot, self.policy = cfg["robot"], policy
    self.default = np.asarray(robot["default_q"])
    self.scale = np.asarray(robot["action_scale"])
    hard = np.asarray(robot["joint_limits"])
    centre, half = hard.mean(axis=1), (hard[:, 1] - hard[:, 0]) / 2
    factor = cfg["hitter"]["soft_limit_factor"]
    self.lower, self.upper = centre - factor * half, centre + factor * half
    self.reset()

  def reset(self):
    self.previous_joint = np.zeros(29, dtype=np.float32)
    self.debug = {}

  def step(self, state, station, position, velocity, tts):
    obs = observation(
      state, self.previous_joint, self.default, station, position, velocity, tts
    )
    action = self.policy(obs)
    target = np.clip(self.default + self.scale * action, self.lower, self.upper)
    self.previous_joint = action.copy()
    self.debug = dict(
      actor_obs=obs,
      joint_action=action,
      target_q=target,
      time_to_strike=tts,
      goal_position=np.asarray(position),
      goal_velocity=np.asarray(velocity),
      goal_base_xy=np.asarray(station),
    )
    return target


class HitterGeometry:
  """Training boxes and strike clock, expressed in the hope_table world."""

  def __init__(self, cfg, options):
    contract = cfg["hitter"]
    dt = cfg["control_dt"]
    self.home_station = np.asarray(options["station"], dtype=float)
    if self.home_station.shape != (2,) or not np.isfinite(self.home_station).all():
      raise ValueError("hitter.station must be a finite [x, y] in hope_table")
    # Training floor is z=0; hope_table puts the floor at the bridge's z.
    self.floor_z = float(cfg["simulation_world_translation"][2])
    self.boxes = np.asarray(contract["racket_pos_range_per_clip"], dtype=float)
    self.velocity_boxes = np.asarray(contract["racket_vel_range_per_clip"], dtype=float)
    clips = contract["clips"]
    if [c["side"] for c in clips] != list(SIDES) or self.boxes.shape != (2, 3, 2):
      raise ValueError("Expected forehand then backhand hitter clips")
    self.windup = np.array([c["strike_frame"] * dt for c in clips])
    self.follow = np.array([(c["frames"] - 1 - c["strike_frame"]) * dt for c in clips])
    self.follow[:] = 0.5
    low, high = contract["base_target_y_range"]
    self.station_y = (self.home_station[1] + low, self.home_station[1] + high)
    self.max_station_step = float(high)

  def plane_x(self, side):
    return float(self.home_station[0] + self.boxes[side, 0].mean())

  def common_x_range(self):
    """One physical planner plane can serve both sides inside this overlap."""
    return self.home_station[0] + np.array(
      [self.boxes[:, 0, 0].max(), self.boxes[:, 0, 1].min()]
    )

  def centre(self, side, station):
    c = self.boxes[side].mean(axis=1)
    return np.array([station[0] + c[0], station[1] + c[1], self.floor_z + c[2]])

  def evaluate(self, side, command, station, now):
    """Mirror play_1v0's RandomServer.plan gates for one planner command."""
    remaining = command.deadline - now
    target = np.asarray(command.position, dtype=float)
    box = self.boxes[side]
    reason = None
    if not command.diagnostic.get("valid", True):
      reason = "planner_invalid_command"
    elif not 0.04 < remaining < 2.5:
      reason = "time_window"
    elif not box[0, 0] <= target[0] - self.home_station[0] <= box[0, 1]:
      reason = "x_outside_training_box"
    elif not box[2, 0] <= target[2] - self.floor_z <= box[2, 1]:
      reason = "height_outside_training_box"
    new_y = float(np.clip(station[1], target[1] - box[1, 1], target[1] - box[1, 0]))
    if reason is None and (
      abs(new_y - station[1]) > self.max_station_step
      or not self.station_y[0] <= new_y <= self.station_y[1]
    ):
      reason = "station_outside_workspace"
    bounds = self.velocity_boxes[side]
    velocity = np.clip(command.velocity, bounds[:, 0], bounds[:, 1])
    diagnostic = dict(
      command.diagnostic,
      side=SIDES[side],
      remaining=remaining,
      plane_x=self.plane_x(side),
      station_y=new_y,
      velocity_clip_mps=float(np.linalg.norm(velocity - command.velocity)),
      reason=reason or "accepted",
    )
    if reason is not None:
      return None, diagnostic
    station = np.array([station[0], new_y])
    shot = Shot(side, target.copy(), velocity, command.deadline, station, diagnostic)
    return shot, diagnostic


class HitterController:
  """Shared-runtime engine: one always-running actor, no reference tracking.

  ARMED feeds the side's training-box centre with the frozen wind-up time to
  strike. An accepted planner command starts EXECUTE; the clock runs down to
  -follow, then ARMED returns to the captured anchor while accepting new shots.
  """

  def __init__(
    self,
    deployment,
    runtime,
    options,
    *,
    home_index=0,
    allow=None,
    selection="fixed",
    receiving=True,
    auto_start=False,
    auto_arm=False,
  ):
    self.cfg, self.runtime, self.options = deployment.settings, runtime, options
    if self.cfg["task"] != "HOPEHitterPingPongG1":
      raise ValueError("HitterController requires the 104-D hitter task")
    hitter = options["hitter"]
    self.geometry = HitterGeometry(self.cfg, hitter)
    self.receiving = receiving
    default_q = np.asarray(self.cfg["robot"]["default_q"])
    # The shared START/preposition path reads library.motions[home].q[0].
    self.home = SimpleNamespace(q=default_q[None].copy())
    self.library = SimpleNamespace(motions=[self.home], paths=deployment.motions)
    self.home_index = self.motion_index = 0
    self.candidate_indices = [0]
    self.lifecycle = Lifecycle(default_q, runtime["transition_seconds"])
    self.controller = HitterActor(self.cfg, deployment.actor)
    self.standing = HopeStanding(self.controller, runtime, self.home)
    self.side_selection = hitter.get("side_selection", "backhand")
    if self.side_selection not in ("auto", *SIDES):
      raise ValueError("hitter.side_selection must be auto, forehand or backhand")
    # Auto starts in the established backhand idle posture. The side is selected
    # from reachable targets when a shot begins, then locked through follow-through.
    self.idle_side = 0 if self.side_selection == "forehand" else 1
    self.side = self.idle_side
    self.return_anchor = None
    self.return_tolerance = float(hitter.get("return_tolerance_m", 0.05))
    if not np.isfinite(self.return_tolerance) or self.return_tolerance <= 0:
      raise ValueError("hitter.return_tolerance_m must be finite and positive")
    self.external_planner_command = None
    self.external_planner_received_at = None
    self.external_planner_deadline = None  # Fixed strike deadline computed once
    self.external_planner_error = "no_valid_external_command"
    self.completed = self.recovered = 0
    self.events, self.debug, self.command_debug = [], {}, {}
    self.previous_target = self.planner = self.physical_limit_observer = None
    self.command_policy = "hope_hitter_actor"
    self.reset_state()

  def reset_state(self):
    self.side = self.idle_side
    self.station = (
      self.geometry.home_station.copy()
      if self.return_anchor is None else self.return_anchor[:2].copy()
    )
    self.shot = None
    self.returning = False
    self.return_pending = False

  def capture_return_anchor(self, root_pos, now, *, source="policy_entry"):
    """Capture once per controller lifetime, including across DOWN/UP cycles.

    The actor commands base XY only. Keep measured Z for diagnostics, but keep
    racket height in the calibrated table/floor world rather than adding base Z.
    """
    if self.return_anchor is not None:
      return
    position = np.asarray(root_pos, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
      raise ValueError("Hitter return anchor requires a valid measured base XYZ")
    self.return_anchor = position.copy()
    self.station = position[:2].copy()
    self.events.append(dict(
      time=now, event="return_anchor_captured", source=source,
      position=self.return_anchor.tolist(),
    ))
    print(f"[hope_g1_hitter] return anchor={self.return_anchor.tolist()} ({source})", flush=True)

  @property
  def phase(self):
    if self.lifecycle.state == "ARMED" and self.returning:
      return "RETURN"
    return {"ARMED": "IDLE", "EXECUTE": "SWING"}.get(
      self.lifecycle.state, self.lifecycle.state
    )

  def describe(self):
    g = self.geometry
    return (
      f"[hope_g1_hitter] task={self.cfg['task']}; actor=104->29; "
      f"side_selection={self.side_selection}; "
      f"station={g.home_station.tolist()}; "
      f"planes FH={g.plane_x(0):.3f} BH={g.plane_x(1):.3f}; "
      f"common_x_range={g.common_x_range().tolist()}; "
      f"return_anchor=first_UP; return_tolerance={self.return_tolerance:.3f}m; "
      f"windup={g.windup.tolist()}; follow={g.follow.tolist()}"
    )

  def candidates(self, state):
    return []

  def update_ball_observations(self, observations):
    # External ROS planner receives ball data directly; no update needed here.
    pass

  def update_external_planner_command(self, packet, received):
    """Use the same ROS-to-RacketCommand conversion as the real HOPE policy."""
    if packet is None:
      return
    received = float(received)
    self.external_planner_command, self.external_planner_error = (
      decode_ros_planner_command(packet, received)
    )
    self.external_planner_received_at = received
    command = self.external_planner_command
    self.external_planner_deadline = command.deadline if command is not None else None

  def start_policy(self, target, now, *, reset_history=False, blend_reference=True):
    if self.lifecycle.state not in ("STABLE", "POSE_HOLD"):
      raise RuntimeError("Hitter policy entry requires STABLE or POSE_HOLD")
    self.controller.reset()
    self.reset_state()
    self.lifecycle.state = "ARMED"
    self.previous_target = np.asarray(target).copy()
    self.events.append(dict(time=now, event="hope_hitter_policy_enabled"))

  def reset_manual_policy(self, now):
    self.controller.reset()
    self.reset_state()
    self.lifecycle.state = "WAIT"
    self.events.append(dict(time=now, event="returned_to_stable"))

  def select(self, now, state=None):
    """Pick the reachable side requiring least lateral travel from the robot.

    Preserve the planner's position and deadline: neither can be moved to a
    different X plane without replanning the incoming ball trajectory.
    """
    command = self.external_planner_command
    if command is None:
      diagnostic = dict(side=SIDES[self.side], reason=self.external_planner_error)
      return None, [diagnostic]

    sides = [self.side]
    if self.side_selection == "auto" and self.shot is None:
      sides.append(1 - self.side)
    reference_y = self.station[1] if state is None else state.root_pos[1]
    selection_station = self.station
    if self.shot is None and state is not None:
      # A return target is not the robot's current position. Evaluate the next
      # shot's lateral step from measured Y, including while returning. Strike
      # station X and workspace remain in the existing table calibration.
      selection_station = np.array([self.geometry.home_station[0], reference_y])
    attempts, candidates = [], []
    for side in sides:
      shot, diagnostic = self.geometry.evaluate(side, command, selection_station, now)
      diagnostic["selected"] = False
      if shot is not None:
        travel = abs(shot.station[1] - reference_y)
        diagnostic["lateral_travel_m"] = float(travel)
        # Keep the previous side on ties. Velocity clipping is not a substitute
        # for reaching the target position.
        candidates.append(((travel, side != self.side), shot))
      attempts.append(diagnostic)
    if not candidates:
      return None, attempts
    shot = min(candidates, key=lambda item: item[0])[1]
    shot.diagnostic["selected"] = True
    return shot, attempts

  def step(self, state, estimate, now, *, racket_command=None):
    # racket_command is handled via update_external_planner_command, not here
    if self.lifecycle.state == "HALT":
      return None
    if self.return_anchor is None and self.lifecycle.state in ("ARMED", "EXECUTE"):
      # Direct/offline callers without the shared operator hook capture at the
      # first policy observation. Real IO captures at the UP edge, before posture.
      self.capture_return_anchor(state.root_pos, now)
    self.standing.observe(state)
    if self.standing.debug["tilt_degrees"] > self.runtime["max_tilt_degrees"]:
      error = ValueError("Hitter robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("hope_tilt", error)
    g = self.geometry
    if (
      self.lifecycle.state == "EXECUTE"
      and now - self.shot.strike_at > g.follow[self.side]
    ):
      self.completed += 1
      self.shot = None
      self.station = self.return_anchor[:2].copy()
      self.return_pending = True
      self.lifecycle.state = "ARMED"
      self.events.append(dict(time=now, event="swing_finished", side=SIDES[self.side]))
      self.events.append(dict(
        time=now, event="return_to_anchor_started",
        anchor_xy=self.return_anchor[:2].tolist(),
      ))
    decision = {"reason": "waiting_for_ball"}
    # play_1v0 re-accepts every reachable solve, including a new ball arriving
    # during follow-through; expired commands are dropped by the planners.
    if self.receiving and self.lifecycle.state in ("ARMED", "EXECUTE"):
      shot, attempts = self.select(now, state)
      decision = dict(reason="accepted" if shot else "no_reachable_command")
      decision["attempts"] = attempts
      if shot is not None:
        if self.lifecycle.state == "ARMED":
          if self.return_pending:
            self.events.append(dict(time=now, event="return_interrupted_by_command"))
          self.return_pending = False
          self.lifecycle.state = "EXECUTE"
          self.events.append(dict(time=now, event="motion_selected", **shot.diagnostic))
        self.shot, self.side, self.station = shot, shot.side, shot.station
    anchor_error = (
      None if self.return_anchor is None
      else float(np.linalg.norm(state.root_pos[:2] - self.return_anchor[:2]))
    )
    self.returning = False
    if self.shot is not None:
      position, velocity = self.shot.position, self.shot.velocity
      remaining = self.shot.strike_at - now
      tts = float(np.clip(remaining, -g.follow[self.side], g.windup[self.side]))
    else:
      if self.return_anchor is not None:
        self.station = self.return_anchor[:2].copy()
        self.returning = anchor_error > self.return_tolerance
        if self.return_pending and not self.returning:
          self.recovered += 1
          self.return_pending = False
          self.events.append(dict(
            time=now, event="return_anchor_reached", error_m=anchor_error,
          ))
      position = g.centre(self.side, self.station)
      velocity = g.velocity_boxes[self.side].mean(axis=1)
      tts = float(g.windup[self.side])
    target = self.controller.step(state, self.station, position, velocity, tts)
    self.motion_index = 0
    self.planner = SimpleNamespace(diagnostic=decision)
    self.command_debug = dict(
      self.controller.debug,
      side=SIDES[self.side],
      swing_active=self.shot is not None,
      return_anchor_world=(
        None if self.return_anchor is None else self.return_anchor.copy()
      ),
      return_anchor_error_m=anchor_error,
      returning_to_anchor=self.returning,
      decision=decision,
    )
    self.debug = dict(self.command_debug, decision=decision)
    self.previous_target = target.copy()
    return target

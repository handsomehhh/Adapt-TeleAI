"""Continuous 104-D G1 hitter controller, ported from hope_mjlab play_1v0.

World: hope_table (tabletop Z=0). Training world has the same axes; its origin
is the robot station on the floor, so every target crosses only a translation.
"""

import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from deploy.g1_pingpong.core import rotation
from deploy.g1_pingpong.lifecycle import Lifecycle
from deploy.hope_g1.controller import HopeStanding
from deploy.racket_command import decode_g1_planner_command

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
    self.follow = np.full(2, 0.5)
    self.station_y = (-1.25, -0.35)
    self.max_station_step = 0.35

  def plane_x(self, side):
    return float(self.home_station[0] + self.boxes[side, 0].mean())

  def centre(self, side, station):
    c = self.boxes[side].mean(axis=1)
    return np.array([station[0] + c[0], station[1] + c[1], self.floor_z + c[2]])

  def evaluate(self, side, command, station, now):
    """Validate and execute the planner's complete solve, without modifying it."""
    remaining = command.deadline - now
    target = np.asarray(command.position, dtype=float)
    requested_station = np.asarray(command.diagnostic["station"], dtype=float)
    box = self.boxes[side]
    reason = None
    if not command.diagnostic.get("valid", True):
      reason = "planner_invalid_command"
    # The planner already checks 0.04 < TTS < 2.5 at the observation time.
    # Transport delay must not add a second 40 ms cutoff at the actor.
    elif not 0 < remaining < 2.5:
      reason = "time_window"
    elif not box[2, 0] <= target[2] - self.floor_z <= box[2, 1]:
      reason = "height_outside_training_box"
    if (
      reason is None
      and not self.station_y[0] <= requested_station[1] <= self.station_y[1]
    ):
      reason = "station_outside_workspace"
    if (
      reason is None
      and abs(requested_station[1] - station[1]) > self.max_station_step + 1e-7
    ):
      reason = "station_too_far_from_measured_y"
    bounds = self.velocity_boxes[side]
    velocity = np.asarray(command.velocity)
    if reason is None and (
      np.any(velocity < bounds[:, 0] - 1e-7) or np.any(velocity > bounds[:, 1] + 1e-7)
    ):
      reason = "velocity_outside_training_box"
    relative_xy = target[:2] - requested_station
    if reason is None and (
      np.any(relative_xy < box[:2, 0] - 1e-7) or np.any(relative_xy > box[:2, 1] + 1e-7)
    ):
      reason = "position_outside_training_box"
    normal = velocity / max(np.linalg.norm(velocity), 1e-9)
    if reason is None and not np.allclose(command.normal, normal, atol=1e-6):
      reason = "normal_velocity_mismatch"
    diagnostic = dict(
      command.diagnostic,
      side=SIDES[side],
      remaining=remaining,
      plane_x=self.plane_x(side),
      station_y=float(requested_station[1]),
      velocity_clip_mps=0.0,
      reason=reason or "accepted",
    )
    if reason is not None:
      return None, diagnostic
    shot = Shot(
      side,
      target.copy(),
      velocity.copy(),
      command.deadline,
      requested_station.copy(),
      diagnostic,
    )
    return shot, diagnostic


class HitterController:
  """Shared-runtime engine: one always-running actor, no reference tracking.

  ARMED feeds the side's training-box centre with the frozen wind-up time to
  strike. An accepted planner command starts EXECUTE; the clock runs down to
  -0.5 seconds, then the idle target returns to the first-UP measured anchor.
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
    self.return_anchor = None
    self.return_tolerance = float(hitter.get("return_tolerance_m", 0.05))
    if not np.isfinite(self.return_tolerance) or self.return_tolerance <= 0:
      raise ValueError("hitter.return_tolerance_m must be finite and positive")
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
    self.contract_hash = self.cfg["hitter_contract_sha256"]
    self.feedback_sequence = 0
    self.external_planner_command = None
    self.external_planner_received_at = None
    self.external_planner_deadline = None  # Fixed strike deadline computed once
    self.external_planner_error = "no_valid_external_command"
    self.completed = self.recovered = 0
    self.events, self.debug, self.command_debug = [], {}, {}
    self.previous_target = self.planner = self.physical_limit_observer = None
    self.command_policy = "hope_hitter_actor"
    from pathlib import Path

    import yaml
    from hope_planner_g1.profiles import load_profile

    params = yaml.safe_load(Path(hitter["planner_config"]).read_text())[
      "hope_planner_g1"
    ]["ros__parameters"]
    self.runtime_mode = options.get(
      "runtime_mode",
      "sim" if options.get("planner_transport") == "embedded" else "real",
    )
    self.physics_profile = load_profile(
      self.runtime_mode, params.get("real_config") or None
    )
    self.embedded_observer = None
    if options.get("planner_transport") == "embedded":
      from hope_planner_g1.observation import BallObserver
      from hope_planner_g1.planner import G1Planner

      planner = G1Planner(
        self.cfg["hitter"],
        mode=self.runtime_mode,
        real_config=params.get("real_config") or None,
        landing=params["landing_xy"],
        flight_time=params["return_flight_time"],
      )
      self.embedded_observer = BallObserver(planner)
    self.reset_state()

  def reset_state(self):
    # Same initial side as MultiPlanner's HomeController.
    reach = self.geometry.home_station[1] + self.geometry.boxes[0, 1].mean()
    self.side = 0 if -1.525 + 0.12 < reach < -0.12 else 1
    self.station = (
      self.geometry.home_station.copy()
      if self.return_anchor is None
      else self.return_anchor[:2].copy()
    )
    self.planning_station = np.array([self.geometry.home_station[0], self.station[1]])
    self.return_pending = self.returning = False
    self.shot = None
    self.session_id = uuid.uuid4().hex
    self.last_command_id = -1
    self.external_planner_command = None
    self.external_planner_error = "waiting_for_current_session_command"
    if self.embedded_observer is not None:
      self.embedded_observer.reset()

  def capture_return_anchor(self, root_pos, now, *, source="policy_entry"):
    """Keep the first measured anchor across DOWN/UP; Z is diagnostic only."""
    if self.return_anchor is not None:
      return
    position = np.asarray(root_pos, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
      raise ValueError("Hitter return anchor requires a valid measured base XYZ")
    self.return_anchor = position.copy()
    self.station = position[:2].copy()
    self.planning_station = np.array([self.geometry.home_station[0], position[1]])
    self.events.append(
      dict(
        time=now,
        event="return_anchor_captured",
        source=source,
        position=position.tolist(),
      )
    )
    print(f"[hope_g1_hitter] return anchor={position.tolist()} ({source})", flush=True)

  def planner_state(self):
    self.feedback_sequence += 1
    return dict(
      runtime_mode=self.runtime_mode,
      physics_sha256=self.physics_profile.source_sha256,
      contract_sha256=self.contract_hash,
      session_id=self.session_id,
      sequence=self.feedback_sequence,
      enabled=bool(self.receiving and self.lifecycle.state in ("ARMED", "EXECUTE")),
      side=self.side,
      # Keep the calibrated hit-plane X; reachability starts at measured Y.
      station=self.planning_station.tolist(),
      home=self.geometry.home_station.tolist(),
    )

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
      f"mode={self.runtime_mode}; physics={self.physics_profile.paddle_model}; "
      f"station={g.home_station.tolist()}; "
      f"return_anchor=first_UP; return_tolerance={self.return_tolerance:.3f}m; "
      f"planes FH={g.plane_x(0):.3f} BH={g.plane_x(1):.3f}; "
      f"windup={g.windup.tolist()}; follow={g.follow.tolist()}"
    )

  def candidates(self, state):
    return []

  def update_ball_observations(self, observations):
    if self.embedded_observer is not None:
      for _stamp, position, source_monotonic in observations:
        # The receiver maps acquisition time into the control clock. Keep
        # source intervals instead of using the policy callback timestamp.
        self.embedded_observer.observe(source_monotonic, position)

  def reset_ball_observations(self):
    """A simulator spawn starts a new track, without resetting actor or shot.

    Live ROS input uses its own position-stream gap/direction detection. This
    explicit boundary is only used when the simulator replaces its ball.
    """
    if self.embedded_observer is not None:
      self.embedded_observer.reset()

  def update_external_planner_command(self, packet, received):
    """Use the same ROS-to-RacketCommand conversion as the real HOPE policy."""
    if packet is None:
      return
    try:
      received = float(received)
      command_id = int(packet.get("command_id", -1))
    except (TypeError, ValueError, OverflowError):
      self.external_planner_command = None
      self.external_planner_error = "malformed_g1_command"
      return
    if (
      packet.get("session_id") != self.session_id or command_id <= self.last_command_id
    ):
      return
    self.last_command_id = command_id
    self.external_planner_command, self.external_planner_error = (
      decode_g1_planner_command(
        packet,
        received,
        contract_hash=self.contract_hash,
        session_id=self.session_id,
        runtime_mode=self.runtime_mode,
        physics_sha256=self.physics_profile.source_sha256,
      )
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

  def select(self, now):
    """Consume each solve once; retain an accepted shot through follow-through."""
    command = self.external_planner_command
    self.external_planner_command = None
    if command is None:
      diagnostic = dict(side=SIDES[self.side], reason=self.external_planner_error)
      return None, [diagnostic]
    if now - self.external_planner_received_at > 0.2:
      return None, [dict(reason="planner_transport_stale")]
    side = command.diagnostic["side"]
    shot, diagnostic = self.geometry.evaluate(side, command, self.planning_station, now)
    return shot, [diagnostic]

  def step(self, state, estimate, now, *, racket_command=None):
    # racket_command is handled via update_external_planner_command, not here
    if self.lifecycle.state == "HALT":
      return None
    if self.return_anchor is None:
      # Shared real/MuJoCo IO captures on UP; direct callers capture here.
      self.capture_return_anchor(state.root_pos, now)
    self.planning_station = np.array([self.geometry.home_station[0], state.root_pos[1]])
    self.standing.observe(state)
    if self.standing.debug["tilt_degrees"] > self.runtime["max_tilt_degrees"]:
      error = ValueError("Hitter robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("hope_tilt", error)
    g = self.geometry
    if self.embedded_observer is not None and self.receiving:
      command = self.embedded_observer.update(now, self.planning_station, self.side)
      if command is not None:
        self.update_external_planner_command(
          dict(
            schema="g1_1v1_v1",
            runtime_mode=self.runtime_mode,
            physics_sha256=self.physics_profile.source_sha256,
            contract_sha256=self.contract_hash,
            session_id=self.session_id,
            command_id=self.last_command_id + 1,
            state_sequence=self.feedback_sequence,
            valid=True,
            side=command.side,
            station=command.station.tolist(),
            position=command.position.tolist(),
            velocity=command.velocity.tolist(),
            normal=command.normal.tolist(),
            bridge_monotonic=now,
            time_to_strike=command.strike_time - now,
            source_ball_time=command.source_time,
            predicted_bounces=command.bounces,
          ),
          now,
        )
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
      self.events.append(
        dict(
          time=now,
          event="return_to_anchor_started",
          anchor_xy=self.return_anchor[:2].tolist(),
        )
      )
    decision = {"reason": "waiting_for_ball"}
    # play_1v0 re-accepts every reachable solve, including a new ball arriving
    # during follow-through; expired commands are dropped by the planners.
    if self.receiving and self.lifecycle.state in ("ARMED", "EXECUTE"):
      shot, attempts = self.select(now)
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
    anchor_error = float(np.linalg.norm(state.root_pos[:2] - self.return_anchor[:2]))
    self.returning = False
    if self.shot is not None:
      position, velocity = self.shot.position, self.shot.velocity
      remaining = self.shot.strike_at - now
      tts = float(np.clip(remaining, -g.follow[self.side], g.windup[self.side]))
    else:
      self.station = self.return_anchor[:2].copy()
      self.returning = anchor_error > self.return_tolerance
      if self.return_pending and not self.returning:
        self.recovered += 1
        self.return_pending = False
        self.events.append(
          dict(time=now, event="return_anchor_reached", error_m=anchor_error)
        )
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
      return_anchor_world=self.return_anchor.copy(),
      return_anchor_error_m=anchor_error,
      returning_to_anchor=self.returning,
      decision=decision,
    )
    self.debug = dict(self.command_debug, decision=decision)
    self.previous_target = target.copy()
    return target

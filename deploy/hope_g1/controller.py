"""HOPE single-actor receiving controller; no Stage 1/2 policy dependencies."""

from dataclasses import dataclass

import numpy as np

from deploy.g1_pingpong.core import quaternion_multiply, rotation
from deploy.g1_pingpong.lifecycle import Lifecycle
from deploy.g1_pingpong.motions import MotionLibrary, RebasedMotion
from deploy.g1_pingpong.perception import ReceivePlanner
from deploy.hope_g1.a3_planner import ReceivePlan, command_goal
from deploy.racket_command import RacketCommand


@dataclass
class RacketGoal:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  base_xy: np.ndarray


class HopeActorController:
  """Construct the saved 175-D actor observation in Isaac articulation order."""

  def __init__(self, cfg, policy):
    self.cfg, self.policy = cfg, policy
    self.default = np.asarray(cfg["robot"]["default_q"])
    self.default_dq = np.asarray(cfg["robot"]["default_dq"])
    self.scale = np.asarray(cfg["robot"]["action_scale"])
    self.reset()

  def reset(self):
    self.previous_joint = np.zeros(29, dtype=np.float32)
    self.previous_position = self.previous_time = None
    self.motion = None
    self.t = 0.0
    self.debug = {}

  def observation(self, state, reference, frame, goal, time_to_strike, now):
    frame = min(max(int(frame), 0), len(reference.q) - 1)
    q, pos, quat = reference.sample(frame / reference.fps)
    anchor_inv = rotation(state.anchor_quat).T
    root_inv = rotation(state.root_quat).T
    root_rot = rotation(state.root_quat)
    yaw = np.arctan2(root_rot[1, 0], root_rot[0, 0])
    heading_inv = rotation([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]).T
    # The actor was trained on world-frame targets in the NPZ/training world.
    # A live world rebase rotates the robot, reference and physical targets;
    # map these two absolute-vector terms back into that training world.
    policy_world_inv = reference.rotation.T
    # Use measured odometry, also in simulation; no plant-velocity oracle.
    velocity = np.zeros(3)
    if hasattr(state, "root_velocity_world"):
      velocity = np.asarray(state.root_velocity_world).copy()
    elif self.previous_time is not None:
      elapsed = now - self.previous_time
      if elapsed <= 0:
        raise ValueError("HOPE observation time must increase")
      velocity = (state.root_pos - self.previous_position) / elapsed
    terms = [
      q,
      reference.dq[frame],
      anchor_inv @ (pos - state.anchor_pos),
      (anchor_inv @ rotation(quat))[:, :2].reshape(-1),
      root_inv @ velocity,
      state.gyro,
      state.q - self.default,
      state.dq - self.default_dq,
      self.previous_joint,
      root_inv @ [0.0, 0.0, -1.0],
      (heading_inv @ np.r_[goal.base_xy - state.root_pos[:2], 0.0])[:2],
      heading_inv @ (goal.position - state.root_pos),
      policy_world_inv @ goal.velocity,
      policy_world_inv @ goal.normal,
      [time_to_strike],
    ]
    observation = np.concatenate(terms).astype(np.float32)
    if observation.shape != (175,) or not np.isfinite(observation).all():
      raise ValueError("Invalid HOPE observation")
    return observation, velocity

  def step(self, state, reference, frame, goal, time_to_strike, now):
    obs, velocity = self.observation(state, reference, frame, goal, time_to_strike, now)
    action = self.policy(obs)
    target = self.default + self.scale * action
    if target.shape != (29,) or not np.isfinite(target).all():
      raise ValueError("Invalid HOPE target")
    self.previous_joint = action.copy()
    self.previous_position, self.previous_time = state.root_pos.copy(), now
    self.motion, self.t = reference, frame / reference.fps
    self.debug = dict(
      actor_obs=obs,
      joint_action=action,
      target_q=target,
      reference_time=self.t,
      reference_frame=frame,
      time_to_strike=time_to_strike,
      root_velocity_world=velocity,
      goal_position=goal.position,
      goal_velocity=goal.velocity,
      goal_normal=goal.normal,
      goal_velocity_policy_frame=obs[168:171].copy(),
      goal_normal_policy_frame=obs[171:174].copy(),
      goal_base_xy=goal.base_xy,
    )
    return target


class HopeStanding:
  """Readiness diagnostics consumed by the shared AMP handoff."""

  def __init__(self, actor, runtime, home):
    self.controller, self.runtime, self.home = actor, runtime, home
    self.readiness_mode = runtime.get("readiness_mode", "timed")
    self.previous = None
    self.debug = {}

  def observe(self, state):
    limits = self.runtime["standing"]
    metrics = dict(
      pose_error=float(np.max(np.abs(state.q - self.home.q[0]))),
      joint_velocity=float(np.max(np.abs(state.dq))),
      gyro=float(np.max(np.abs(state.gyro))),
      tilt_degrees=float(
        np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
      ),
    )
    settled = all(
      v <= limits[f"max_{k}"] for k, v in metrics.items() if f"max_{k}" in limits
    )
    self.debug = {**metrics, "stable": settled, "readiness_mode": self.readiness_mode}
    return settled, metrics


class HopeController:
  """A3-compatible triggered receiving around a single G1 HOPE actor.

  Goals latch at selection. Idle/return use frame zero and the same actor;
  the motion clock advances only during EXECUTE. No tracker is substituted.
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
    self.target_source = options.get("target_source", "reference")
    if self.target_source == "a3_planner":
      # Keep old command lines/configs usable while routing them to the exact
      # current ROS planner core rather than the previous local A3 clone.
      self.target_source = "hope_planner_optimized_2"
    if self.target_source not in ("reference", "hope_planner_optimized_2"):
      raise ValueError("target_source must be reference or hope_planner_optimized_2")
    self.planner_active = self.target_source == "hope_planner_optimized_2"
    self.planner_transport = options.get("planner_transport", "embedded")
    if self.planner_transport not in ("embedded", "ros2"):
      raise ValueError("planner_transport must be embedded or ros2")
    if self.planner_transport == "ros2" and not self.planner_active:
      raise ValueError("ROS planner transport requires target_source=hope_planner_optimized_2")
    self.a3_options = options.get("a3", {})
    self.target_mode = self.a3_options.get("target_mode", "raw")
    self.reference_mode = self.a3_options.get("reference_mode", "a3_live")
    if self.target_mode not in ("raw", "envelope") or self.reference_mode not in (
      "a3_live",
      "training_common",
    ):
      raise ValueError("Invalid A3 target_mode/reference_mode")
    self.command_planner = None
    if self.planner_active and self.planner_transport == "embedded":
      from deploy.hope_g1.hope_planner_adapter import HopePlannerAdapter

      self.command_planner = HopePlannerAdapter(
        hit_plane_x=self.a3_options.get("hit_plane_x"),
        config_path=options.get("hope_planner_config"),
      )
    self.external_planner_command = None
    self.external_planner_received = None
    self.external_planner_error = "waiting_for_ros_planner_command"
    self.select_ready_time = 0.0
    self.raw_command = None
    if self.cfg["task"] != "HOPEPingPongMultiMotionG1":
      raise ValueError("HopeController requires the independent HOPE task")
    if selection not in ("fixed", "auto"):
      raise ValueError("HOPE motion selection must be fixed or auto")
    if (
      not np.isfinite(options["max_intercept_distance"])
      or options["max_intercept_distance"] <= 0
      or not 0 <= options["max_start_frame"] <= self.cfg["hope"]["max_start_frame"]
    ):
      raise ValueError("Invalid HOPE interception distance/start-frame window")
    self.selection, self.receiving = selection, receiving
    self.auto_start, self.auto_arm = auto_start, auto_arm
    self.library = MotionLibrary(
      deployment, [home_index] if selection == "fixed" else allow
    )
    self.library.hit_types = self.cfg["hope"]["hit_types"]
    if not 0 <= home_index < len(self.library.motions):
      raise ValueError("Invalid HOPE home index")
    self.home_index = self.motion_index = home_index
    # The saved task derives strike frames from strike_phase, not NPZ annotations.
    for clip in self.library.motions:
      clip.annotated_strike = (
        round((len(clip.q) - 1) * self.cfg["hope"]["racket"]["strike_phase"]) / clip.fps
      )
    self.home = self.library.motions[home_index]
    self.lifecycle = Lifecycle(self.home.q[0], runtime["transition_seconds"])
    self.controller = HopeActorController(self.cfg, deployment.actor)
    self.standing = HopeStanding(self.controller, runtime, self.home)
    self.completed = self.recovered = 0
    self.events, self.debug = [], {}
    self.previous_target = self.planner = None
    self.physical_limit_observer = None
    self.home_frozen = False
    self.references = None
    self.frame = 0
    self.goal = None
    racket = self.cfg["hope"]["racket"]
    self.target_perturb_scale = options.get(
      "target_perturb_scale", racket["ref_perturb_curriculum_start"]
    )
    if (
      not np.isfinite(self.target_perturb_scale)
      or not 0 < self.target_perturb_scale <= 1
    ):
      raise ValueError("HOPE target perturbation scale must be within (0, 1]")
    self.target_offset_limit = (
      np.asarray(racket["ref_perturb_pos"]) * self.target_perturb_scale
    )
    self.recovery_stable_since = None
    training_return_steps = round(
      self.cfg["hope"]["return_seconds"] / self.cfg["control_dt"]
    )
    return_steps = options.get("return_steps", training_return_steps)
    select_settle_ticks = self.a3_options.get("select_settle_ticks", 25)
    for name, value in (
      ("return_steps", return_steps),
      ("a3.select_settle_ticks", select_settle_ticks),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    self.return_frame_count = return_steps
    self.select_settle_ticks = select_settle_ticks
    self.return_frames_sent = 0
    self.command_debug = {}

  @property
  def candidate_indices(self):
    return [self.home_index] if self.selection == "fixed" else self.library.allow

  @property
  def phase(self):
    return {
      "ARMED": "IDLE",
      "EXECUTE": "SWING",
      "RECOVER": "RETURN",
      "READY": "IDLE",
    }.get(self.lifecycle.state, self.lifecycle.state)

  def describe(self):
    return (
      f"[hope_g1] task={self.cfg['task']}; home={self.home_index} "
      f"({self.library.paths[self.home_index].name}); selection={self.selection}; "
      f"candidates={self.candidate_indices}; actor=175->29; "
      f"target_source={self.target_source}; planner_transport={self.planner_transport}; "
      f"return_steps={self.return_frame_count}; select_settle_ticks={self.select_settle_ticks}"
    )

  def candidates(self, state):
    if self.planner_active and self.reference_mode == "a3_live":
      # A3 idle_rebase_for_motion with --fixed-world-home: each clip's frame
      # zero anchor is placed at the frozen home's ready anchor.
      if self.home_frozen:
        _, position, quaternion = self.home.sample(0)
      else:
        position, quaternion = state.anchor_pos, state.anchor_quat
      return [
        (i, RebasedMotion(self.library.motions[i], position, quaternion))
        for i in self.candidate_indices
      ]
    references = self.references or self.aligned_library(state)
    return [(i, references[i]) for i in self.candidate_indices]

  def aligned_library(self, state):
    """Keep all clips in one fixed training-world frame, as in play.

    Isaac adds the same environment origin to every clip. Recentring each
    clip on the moving robot at selection changes its anchor/target contract.
    """
    home = RebasedMotion(
      self.library.motions[self.home_index], state.anchor_pos, state.anchor_quat
    )
    references = []
    for clip in self.library.motions:
      _, p, q = clip.sample(0)
      references.append(
        RebasedMotion(
          clip,
          home.rotation @ p + home.translation,
          quaternion_multiply(home.quaternion, q),
        )
      )
    return references

  def reference_goal(self, reference, index, *, home=False):
    cfg = self.cfg["hope"]["racket"]
    f = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    source = reference.source
    body = source.body_names.index(cfg["wrist_body_name"])
    wrist_q = source.body_quat[f, body]
    mount_q = quaternion_multiply(wrist_q, cfg["mount_quat"])
    normal = (
      reference.rotation
      @ rotation(mount_q)[:, cfg["mount_normal_axis"]]
      * cfg["mount_normal_sign"]
    )
    normal /= np.linalg.norm(normal) + 1e-6
    sign = (
      -1 if self.library.hit_types[index] in cfg["ref_velocity_backhand_values"] else 1
    )
    position = reference.strike_position(self.cfg["ball"])
    base = reference.body_position("pelvis", 0 if home else f / reference.fps)[:2]
    return RacketGoal(position, normal * sign * cfg["ref_velocity_speed"], normal, base)

  def select(self, state, estimate, now):
    if self.planner_active:
      return self.select_planner_command(state, estimate, now)
    choices = []
    for index, reference in self.candidates(state):
      planner = ReceivePlanner(
        reference,
        self.cfg["ball"],
        spatial_tolerance=[np.inf] * 3,
        timing_tolerance=self.runtime["planner"]["timing_tolerance"],
        max_start_frame=self.options["max_start_frame"],
      )
      if not planner.try_start(estimate, now):
        self.debug["decision"] = {**planner.diagnostic, "motion_index": index}
        continue
      distance = float(
        np.linalg.norm(planner.plan[0] - reference.strike_position(self.cfg["ball"]))
      )
      offset_world = planner.plan[0] - reference.strike_position(self.cfg["ball"])
      # The training perturbation box is axis-aligned in NPZ/Isaac world,
      # exactly like the absolute-vector terms in the actor observation.
      offset = reference.rotation.T @ offset_world
      if np.any(np.abs(offset) > self.target_offset_limit):
        self.debug["decision"] = dict(
          reason="goal_outside_play_perturbation",
          motion_index=index,
          offset=offset.tolist(),
          offset_frame="training_world",
          offset_deploy_world=offset_world.tolist(),
          allowed_offset=self.target_offset_limit.tolist(),
        )
        continue
      if distance <= self.options["max_intercept_distance"]:
        choices.append((distance, index, reference, planner))
    if not choices:
      return None
    return min(choices, key=lambda choice: (choice[0], choice[1]))

  def select_planner_command(self, state, estimate, now):
    if now < self.select_ready_time:
      self.debug["decision"] = dict(
        reason="planner_selection_settle", remaining=self.select_ready_time - now
      )
      return None
    command = self.latest_planner_command(now)
    if command is None:
      diagnostics = (
        self.command_planner.last_update_diagnostics
        if self.command_planner is not None
        else {"reason": self.external_planner_error, "planner_transport": "ros2"}
      )
      self.debug["decision"] = dict(
        diagnostics or {"reason": "planner_has_no_published_command"}
      )
      return None
    return self.select_racket_command(state, command, now)

  def latest_planner_command(self, now):
    if not self.planner_active:
      return None
    if self.planner_transport == "embedded":
      return self.command_planner.latest_command(now)
    command = self.external_planner_command
    received = self.external_planner_received
    timeout = float(self.a3_options.get("command_timeout_seconds", 0.2))
    if command is None or received is None:
      return None
    if now - received > timeout:
      self.external_planner_error = "ros_planner_command_stale"
      return None
    if command.deadline <= now:
      self.external_planner_error = "ros_planner_command_expired"
      return None
    return command

  def update_external_planner_command(self, packet, received):
    """Accept an A30808 ROS command relayed over local UDP from the ROS bridge."""
    if self.planner_transport != "ros2" or packet is None:
      return
    received = float(received)
    if not packet.get("valid", False):
      self.external_planner_command = None
      self.external_planner_received = received
      self.external_planner_error = "ros_planner_published_invalid_command"
      return
    try:
      position = np.asarray(packet["position"], dtype=float)
      velocity = np.asarray(packet["velocity"], dtype=float)
      normal = np.asarray(packet["normal"], dtype=float)
      tts = float(packet["time_to_strike"])
      published = float(packet.get("bridge_monotonic", received))
    except (KeyError, TypeError, ValueError):
      self.external_planner_command = None
      self.external_planner_received = received
      self.external_planner_error = "malformed_ros_planner_command"
      return
    if (
      position.shape != (3,)
      or velocity.shape != (3,)
      or normal.shape != (3,)
      or not np.isfinite(position).all()
      or not np.isfinite(velocity).all()
      or not np.isfinite(normal).all()
      or np.linalg.norm(normal) < 1e-6
      or not np.isfinite(tts)
      or not np.isfinite(received)
      or not np.isfinite(published)
      or tts <= 0
    ):
      self.external_planner_command = None
      self.external_planner_received = received
      self.external_planner_error = "malformed_ros_planner_command"
      return
    # The bridge and deploy processes share the host monotonic clock. Using
    # the bridge timestamp keeps the strike deadline from gaining UDP delay.
    diagnostic = {
      "source": "hope_planner_optimized_2_A30808_ros",
      "source_stamp": float(packet.get("source_ball_time", 0.0)),
      "planner_stamp": float(packet.get("stamp", 0.0)),
      "received_monotonic": received,
      "time_to_strike": tts,
      "position": position.tolist(),
      "velocity": velocity.tolist(),
      "normal": normal.tolist(),
      "outgoing_velocity": packet.get("outgoing_velocity"),
      "valid": True,
      "clears_net": bool(packet.get("clears_net", False)),
      "bypasses_net_posts": bool(packet.get("bypasses_net_posts", False)),
      "num_bounces": int(packet.get("predicted_bounces", 0)),
    }
    self.external_planner_command = RacketCommand(
      position=position,
      velocity=velocity,
      normal=normal,
      deadline=published + tts,
      diagnostic=diagnostic,
    )
    self.external_planner_received = received
    self.external_planner_error = ""

  def update_ball_observations(self, observations):
    """Feed source-rate mocap packets directly to the ROS-equivalent planner."""
    if self.command_planner is not None and observations:
      self.command_planner.update_observations(observations)

  def select_racket_command(self, state, command, now):
    """Select using only a public command, independent of its producer."""
    if now < self.select_ready_time:
      self.debug["decision"] = dict(reason="planner_selection_settle")
      return None
    remaining = command.deadline - now
    maximum = self.a3_options.get("max_command_time_to_strike", 0.5)
    scores = []
    for index in self.candidate_indices:
      reference = (
        RebasedMotion(self.library.motions[index], state.anchor_pos, state.anchor_quat)
        if self.reference_mode == "a3_live"
        else dict(self.candidates(state))[index]
      )
      distance = float(
        np.linalg.norm(command.position - reference.strike_position(self.cfg["ball"]))
      )
      scores.append((distance, index, reference))
    best = min(scores, key=lambda item: (item[0], item[1]))
    current = next((item for item in scores if item[1] == self.motion_index), None)
    if (
      current
      and best[1] != current[1]
      and not best[0] + self.options.get("selection_hysteresis", 0.05) < current[0]
    ):
      best = current
    distance, index, reference = best
    if distance > self.options["max_intercept_distance"]:
      self.debug["decision"] = dict(
        reason="a3_intercept_unreachable",
        distance=distance,
        motion_index=index,
        command=command.diagnostic,
      )
      return None
    # A3 only re-initializes the transform when switching motion. The active
    # home keeps its frozen transform when selected again.
    if self.reference_mode == "a3_live" and index == self.home_index:
      reference = self.home
    lead = reference.strike_time(self.cfg["ball"])
    start = max(0, int(np.floor((lead - remaining) * reference.fps + 0.5)))
    diagnostic = dict(
      command.diagnostic,
      start_frame=start,
      motion_index=index,
      remaining=remaining,
      target_mode=self.target_mode,
    )
    if remaining <= 0 or remaining > maximum or start > self.options["max_start_frame"]:
      self.debug["decision"] = dict(diagnostic, reason="arrival_outside_start_window")
      return None
    self.debug["decision"] = dict(diagnostic, reason="accepted")
    return distance, index, reference, ReceivePlan(command, diagnostic)

  def start_policy(self, target, now, *, reset_history=False, blend_reference=True):
    if self.lifecycle.state not in ("STABLE", "POSE_HOLD"):
      raise RuntimeError("HOPE policy entry requires STABLE or POSE_HOLD")
    self.controller.reset()
    # Preserve the existing entry flow and training's zero initial raw action.
    self.lifecycle.state = "ARMED"
    self.previous_target = np.asarray(target).copy()
    self.home_frozen = False
    self.references = None
    self.frame = 0
    self.return_frames_sent = 0
    self.raw_command = None
    self.select_ready_time = now
    self.events.append(dict(time=now, event="hope_policy_enabled"))

  def reset_manual_policy(self, now):
    self.controller.reset()
    self.home = self.library.motions[self.home_index]
    self.home_frozen = False
    self.references = None
    self.goal = self.planner = None
    self.motion_index = self.home_index
    self.return_frames_sent = 0
    self.lifecycle.state = "WAIT"
    self.events.append(dict(time=now, event="returned_to_stable"))

  def step(self, state, estimate, now, *, racket_command=None):
    if racket_command is not None and (
      not self.planner_active or self.target_mode != "raw"
    ):
      raise ValueError("Direct racket commands require the A3 raw target contract")
    if self.lifecycle.state == "HALT":
      return None
    if not self.home_frozen:
      self.references = self.aligned_library(state)
      self.home = self.references[self.home_index]
      self.standing.home = self.home
      self.home_goal = self.reference_goal(self.home, self.home_index, home=True)
      if self.planner_active:
        # A3 derives idle/return base target from the home strike-frame reach,
        # not from frame-zero pelvis XY.
        self.home_goal = self.reference_goal(self.home, self.home_index)
        self.select_ready_time = (
          now + self.select_settle_ticks * self.cfg["control_dt"]
        )
      self.home_frozen = True
      self.events.append(
        dict(
          time=now,
          event="home_frozen",
          translation=self.home.translation.tolist(),
          quaternion_wxyz=self.home.quaternion.tolist(),
        )
      )
    self.standing.observe(state)
    tilt = self.standing.debug["tilt_degrees"]
    if tilt > self.runtime["max_tilt_degrees"]:
      error = ValueError("HOPE robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("hope_tilt", error)
    self.debug = {"decision": {"reason": "waiting_for_ball"}}
    # Each reference is held for a complete control interval. Transition on
    # the next tick after frame N-1, as MotionCommand does after env.step().
    if self.lifecycle.state == "EXECUTE" and self.frame >= len(self.reference.q):
      self.completed += 1
      self.lifecycle.recover(state.q, now)
      self.recovery_stable_since = None
      self.return_frames_sent = 0
      self.events.append(
        dict(
          time=now,
          event="swing_finished",
          motion_index=self.motion_index,
          final_frame=len(self.reference.q) - 1,
          return_steps=self.return_frame_count,
        )
      )
      if self.planner_active:
        self.motion_index = self.home_index
        self.raw_command = None
        self.select_ready_time = (
          now + self.select_settle_ticks * self.cfg["control_dt"]
        )
    if self.lifecycle.state == "RECOVER":
      settled, _ = self.standing.observe(state)
      self.recovery_stable_since = (
        (now if self.recovery_stable_since is None else self.recovery_stable_since)
        if settled
        else None
      )
      elapsed = now - self.lifecycle.transition_start
      ready = self.return_frames_sent >= self.return_frame_count
      if self.standing.readiness_mode == "measured":
        ready &= (
          self.recovery_stable_since is not None
          and now - self.recovery_stable_since
          >= self.runtime["standing"]["dwell_seconds"]
        )
      if ready:
        self.recovered += 1
        self.motion_index = self.home_index
        self.planner = None
        self.lifecycle.ready()
        self.lifecycle.arm(now, True)
        self.events.append(dict(time=now, event="recovered", home=self.home_index))
      elif elapsed > self.runtime["standing"]["timeout_seconds"]:
        raise TimeoutError("HOPE recovery did not stabilize")
    if (
      self.lifecycle.state == "ARMED"
      and self.receiving
      and (
        estimate is not None
        or racket_command is not None
        or self.latest_planner_command(now) is not None
      )
    ):
      choice = (
        self.select_racket_command(state, racket_command, now)
        if racket_command is not None
        else self.select(state, estimate, now)
      )
      if choice is not None:
        distance, self.motion_index, self.reference, self.planner = choice
        self.frame = int(self.planner.diagnostic["start_frame"])
        self.goal = self.reference_goal(self.reference, self.motion_index)
        # The explicit reference baseline only retargets position. The A3
        # branch below replaces it with the complete latched RacketCommand.
        displacement = self.planner.plan[0] - self.goal.position
        # RacketTargetCommand couples XY base placement to contact position:
        # target_base_xy = target_racket_xy - reference_reach_offset_xy.
        self.goal.base_xy = self.goal.base_xy + displacement[:2]
        self.goal.position = self.planner.plan[0].copy()
        if self.planner_active:
          self.raw_command = self.planner.command
          nominal = self.reference_goal(self.reference, self.motion_index)
          sign = (
            -1
            if self.library.hit_types[self.motion_index]
            in self.cfg["hope"]["racket"]["ref_velocity_backhand_values"]
            else 1
          )
          self.goal = RacketGoal(
            *command_goal(
              self.raw_command,
              nominal,
              self.reference,
              self.cfg["hope"]["racket"],
              sign,
              self.target_mode,
              self.target_perturb_scale,
            )
          )
        self.lifecycle.execute(now)
        self.events.append(
          dict(
            time=now,
            event="motion_selected",
            motion_index=self.motion_index,
            motion=self.library.paths[self.motion_index].name,
            distance=distance,
            start_frame=self.frame,
            deadline=self.planner.plan[1],
            target_source=self.target_source,
            raw_command=self.raw_command.diagnostic if self.raw_command else None,
          )
        )
    executing = self.lifecycle.state == "EXECUTE"
    reference = self.reference if executing else self.home
    frame = self.frame if executing else 0
    # Deployment uses a fixed home reference/goal in return and idle instead
    # of training's random resampling of the next clip at wrap.
    goal = self.goal if executing else self.home_goal
    strike_frame = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    target = self.controller.step(
      state,
      reference,
      frame,
      goal,
      (strike_frame - frame) * self.cfg["control_dt"],
      now,
    )
    self.command_policy = "hope_actor"
    self.command_debug = self.controller.debug
    self.command_debug.update(
      target_source=self.target_source,
      planner_target_mode=self.target_mode if self.planner_active else None,
      raw_racket_command=self.raw_command.diagnostic if self.raw_command else None,
      reference_frame_count=len(reference.q),
      return_frame_index=self.return_frames_sent
      if self.lifecycle.state == "RECOVER"
      else None,
    )
    self.debug.update(self.command_debug)
    self.previous_target = target.copy()
    if executing:
      self.frame += 1
    elif self.lifecycle.state == "RECOVER":
      self.return_frames_sent += 1
    return target

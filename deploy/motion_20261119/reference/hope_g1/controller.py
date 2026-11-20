"""HOPE receiving controller around a replaceable executor.

The controller owns command sources (embedded or ROS planner), the lifecycle,
selection settling, recovery accounting, standing checks and events. Joint
targets come from an executor (deploy.motion_20261119.reference.hope_g1.executor), by default a HOPE
actor with its motions and motion selector (MotionExecutor).
"""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import rotation
from deploy.motion_20261119.reference.g1_pingpong.lifecycle import Lifecycle
from deploy.motion_20261119.reference.hope_g1.executor import executor_class
from deploy.motion_20261119.reference.racket_command import RacketCommand, RacketCommandArray
from deploy.motion_20261119.reference.planner_interface import COMMAND_ARRAY, SINGLE_COMMAND, PlannerInterfaceError, validate_selector_interface


class HopeStanding:
  """Readiness diagnostics consumed by the shared AMP handoff."""

  def __init__(self, controller, runtime, home_pose):
    self.controller, self.runtime = controller, runtime
    self.home_pose = np.asarray(home_pose)
    self.readiness_mode = runtime.get("readiness_mode", "timed")
    self.previous = None
    self.debug = {}

  def observe(self, state):
    limits = self.runtime["standing"]
    metrics = dict(
      pose_error=float(np.max(np.abs(state.q - self.home_pose))),
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
  """A3-compatible triggered receiving: commands in, executor joint targets out.

  The executor latches its goal at selection and optionally accepts live
  updates; the controller manages swing entry and readiness for the next one.
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
    if self.target_source not in ("reference", "hope_planner_optimized_2", "hope_planner_command_array"):
      raise ValueError("target_source must be reference, hope_planner_optimized_2 or hope_planner_command_array")
    self.planner_active = self.target_source != "reference"
    self.planner_transport = options.get("planner_transport", "embedded")
    if self.planner_transport not in ("embedded", "ros2"):
      raise ValueError("planner_transport must be embedded or ros2")
    self.planner_command_type = validate_selector_interface(options)
    if self.planner_transport == "ros2" and not self.planner_active:
      raise ValueError("ROS planner transport requires a planner target_source")
    if self.planner_command_type == COMMAND_ARRAY:
      from deploy.motion_20261119.reference.hope_g1.motion_executor import MotionExecutor
      if not issubclass(executor_class(deployment), MotionExecutor):
        raise PlannerInterfaceError("RacketCommandArray requires the motion executor and distance_command_array selector")
    self.a3_options = options.get("a3", {})
    self.target_mode = self.a3_options.get("target_mode", "raw")
    self.command_planner = None
    if self.planner_active and self.planner_transport == "embedded":
      from deploy.motion_20261119.reference.hope_g1.hope_planner_adapter import HopePlannerAdapter

      self.command_planner = HopePlannerAdapter(
        hit_plane_x=self.a3_options.get("hit_plane_x"),
        config_path=options.get("hope_planner_config"),
      )
    self.external_planner_command = None
    self.external_planner_received = None
    self.external_planner_error = "waiting_for_ros_planner_command"
    self._last_external_array_received = None
    self.select_ready_time = 0.0
    self.raw_command = None
    self.raw_command_input = None
    self.executor = executor_class(deployment)(
      deployment,
      runtime,
      options,
      command_driven=self.planner_active,
      home_index=home_index,
      allow=allow,
      selection=selection,
    )
    if options.get("post_swing_base_target", "home") != "home" and not hasattr(
      self.executor, "post_swing_base_target"
    ):
      raise ValueError("post_swing_base_target requires the motion executor")
    transition = options.get("base_target_transition")
    if transition is not None and not hasattr(self.executor, "base_transition") and (
      not isinstance(transition, dict) or transition.get("mode", "step") != "step"
    ):
      raise ValueError("base_target_transition requires the motion executor")
    self.live_command_updates = self.a3_options.get("live_command_updates", False)
    if not isinstance(self.live_command_updates, bool):
      raise ValueError("a3.live_command_updates must be a boolean")
    if self.live_command_updates and (
      not self.planner_active or self.target_mode != "raw"
      or not callable(getattr(self.executor, "update_command", None))
    ):
      raise ValueError("Live command updates require a supported executor and raw planner commands")
    if self.a3_options.get("live_motion_reselection", False) and not getattr(
      self.executor, "live_motion_reselection", False
    ):
      raise ValueError("a3.live_motion_reselection requires the motion executor")
    self.receiving = receiving
    self.auto_start, self.auto_arm = auto_start, auto_arm
    self.lifecycle = Lifecycle(self.home_pose, runtime["transition_seconds"])
    self.standing = HopeStanding(self.executor, runtime, self.home_pose)
    self.completed = self.recovered = 0
    self.events, self.debug = [], {}
    self.previous_target = self.planner = None
    self.physical_limit_observer = None
    self.prepared = False
    self.recovery_stable_since = None
    # Training's return length, when the deployment records one.
    training_return = self.cfg.get("hope", {}).get("return_seconds")
    return_steps = options.get(
      "return_steps",
      None
      if training_return is None
      else round(training_return / self.cfg["control_dt"]),
    )
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

  # Runtime-facing view of the executor (trace, positioning, feeders).
  @property
  def controller(self):
    return self.executor

  @property
  def home_pose(self):
    return self.executor.home_pose

  @property
  def motion_index(self):
    return self.executor.motion_index

  @property
  def motion_name(self):
    return self.executor.motion_name

  @property
  def motion_files(self):
    return self.executor.motion_files

  @property
  def candidate_indices(self):
    return self.executor.candidate_indices

  def candidates(self, state):
    return self.executor.candidates(state)

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
      f"[hope_g1] task={self.cfg['task']}; {self.executor.describe()}; "
      f"target_source={self.target_source}; planner_transport={self.planner_transport}; "
      f"live_command_updates={self.live_command_updates}; "
      f"return_steps={self.return_frame_count}; select_settle_ticks={self.select_settle_ticks}"
    )

  def select(self, state, estimate, now):
    if self.planner_active:
      return self.select_planner_command(state, estimate, now)
    choice = self.executor.choose_from_ball(state, estimate, now)
    if self.executor.decision is not None:
      self.debug["decision"] = self.executor.decision
    return choice

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
    """Enforce the wire contract before decoding single or batched commands."""
    if self.planner_transport != "ros2" or packet is None:
      return
    if packet.get("kind") == "planner_interface_error":
      raise PlannerInterfaceError(packet["error"])
    actual = packet.get("command_type", COMMAND_ARRAY if "commands" in packet else SINGLE_COMMAND)
    if actual != self.planner_command_type:
      raise PlannerInterfaceError(f"Expected {self.planner_command_type}, received {actual}")
    if actual == SINGLE_COMMAND:
      self._update_single_external_planner_command(packet, received)
      return
    # The runtime polls the same UDP slot on every control tick. Preserve batch
    # identity so a cached array cannot repeatedly rewind the active motion.
    if self._last_external_array_received == received:
      return
    self._last_external_array_received = received
    self.external_planner_command = None
    self.external_planner_received = float(received)
    self.external_planner_error = "malformed_ros_planner_command_array"
    try:
      commands, center = packet["commands"], packet["center_index"]
      forward, backward = packet["forward_num_commands"], packet["backward_num_commands"]
      # Counts describe the planner's sampling grid before filtering. The
      # published array may be shorter and its center may move or be absent.
      if (not isinstance(commands, list) or not commands
          or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (forward, backward))
          or isinstance(center, bool) or not isinstance(center, int)
          or not -1 <= center < len(commands)):
        return
      if (not np.isfinite(float(packet["x_hit"]))
          or any(not np.isfinite(float(packet[k])) or float(packet[k]) <= 0
                 for k in ("forward_hit_interval", "backward_hit_interval"))):
        return
      decoded = []
      for index, entry in enumerate(commands):
        if not isinstance(entry, dict):
          self.external_planner_error = "malformed_ros_planner_command_array"
          return
        if not entry.get("valid", False):
          decoded.append(None)
          continue
        # Every deadline uses the batch's bridge clock, never a fresh slot receipt.
        entry = dict(entry, bridge_monotonic=packet.get("bridge_monotonic", received))
        self._update_single_external_planner_command(entry, received)
        cmd = self.external_planner_command
        self.external_planner_command = None
        if cmd is None:
          self.external_planner_error = "malformed_ros_planner_command_array"
          return
        cmd.diagnostic.update(source="ros:/racket/commands", command_index=index, center_index=center)
        decoded.append(cmd)
      metadata = {k: packet[k] for k in (
        "x_hit", "forward_num_commands", "backward_num_commands", "forward_hit_interval", "backward_hit_interval")}
      self.external_planner_command = RacketCommandArray(tuple(decoded), center, metadata)
      self.external_planner_error = ""
    except (KeyError, TypeError, ValueError):
      self.external_planner_command = None
      self.external_planner_error = "malformed_ros_planner_command_array"

  def _update_single_external_planner_command(self, packet, received):
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
      # Any planner node publishing /racket/command; the topic names the source.
      "source": "ros:/racket/command",
      "source_stamp": float(packet.get("source_ball_time", 0.0)),
      "planner_stamp": float(packet.get("stamp", 0.0)),
      "source_ball_position": packet.get("source_ball_position"),
      "stage1_fit_rmse_m": packet.get("stage1_fit_rmse_m"),
      "strike_time": packet.get("strike_time"),
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
    choice = self.executor.choose(state, command, now)
    if choice is not None:
      self.selection_command_input = command
    self.debug["decision"] = self.executor.decision
    return choice

  def start_policy(self, target, now, *, reset_history=False, blend_reference=True):
    if self.lifecycle.state not in ("STABLE", "POSE_HOLD"):
      raise RuntimeError("HOPE policy entry requires STABLE or POSE_HOLD")
    self.executor.reset()
    # Preserve the existing entry flow and training's zero initial raw action.
    self.lifecycle.state = "ARMED"
    self.previous_target = np.asarray(target).copy()
    self.prepared = False
    self.return_frames_sent = 0
    self.raw_command = None
    self.raw_command_input = None
    self.select_ready_time = now
    self.events.append(dict(time=now, event="hope_policy_enabled"))

  def reset_manual_policy(self, now):
    self.executor.reset()
    self.prepared = False
    self.planner = None
    self.return_frames_sent = 0
    self.lifecycle.state = "WAIT"
    self.events.append(dict(time=now, event="returned_to_stable"))

  def step(self, state, estimate, now, *, racket_command=None):
    if racket_command is not None and (
      not self.planner_active or self.target_mode != "raw"
    ):
      raise ValueError("Direct racket commands require the A3 raw target contract")
    if racket_command is not None and (
      isinstance(racket_command, RacketCommandArray) != (self.planner_command_type == COMMAND_ARRAY)
    ):
      raise PlannerInterfaceError(f"Direct input must match {self.planner_command_type}")
    if self.lifecycle.state == "HALT":
      return None
    if not self.prepared:
      frozen = self.executor.prepare(state, now)
      if self.planner_active:
        self.select_ready_time = (
          now + self.select_settle_ticks * self.cfg["control_dt"]
        )
      self.prepared = True
      self.events.append(dict(time=now, event="home_frozen", **frozen))
    self.standing.observe(state)
    tilt = self.standing.debug["tilt_degrees"]
    if tilt > self.runtime["max_tilt_degrees"]:
      error = ValueError("HOPE robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("hope_tilt", error)
    self.debug = {"decision": {"reason": "waiting_for_ball"}}
    if self.live_command_updates and self.receiving and self.lifecycle.state == "EXECUTE":
      command = racket_command if racket_command is not None else self.latest_planner_command(now)
      if command is not None and command is not self.raw_command_input:
        update = self.executor.update_command(state, command, now)
        self.debug["decision"] = self.executor.decision
        if update is not None:
          self.planner = update.plan
          self.raw_command = update.command
          self.raw_command_input = command
          self.events.append(dict(time=now, event="racket_command_updated",
                                  **update.event, deadline=update.command.deadline,
                                  raw_racket_command=update.command.diagnostic))
    if self.lifecycle.state == "EXECUTE" and self.executor.swing_finished(now):
      self.completed += 1
      self.lifecycle.recover(state.q, now)
      self.recovery_stable_since = None
      self.return_frames_sent = 0
      self.events.append(
        dict(
          time=now,
          event="swing_finished",
          **self.executor.finish(state),
          return_steps=self.return_frame_count,
        )
      )
      if self.planner_active:
        self.executor.release()
        self.raw_command = None
        self.raw_command_input = None
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
        self.executor.release()
        self.planner = None
        self.lifecycle.ready()
        self.lifecycle.arm(now, True)
        self.events.append(
          dict(time=now, event="recovered", home=self.executor.motion_index)
        )
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
        swing = self.executor.begin(choice, now)
        self.planner = swing.plan
        if self.planner_active:
          self.raw_command = swing.command
          self.raw_command_input = self.selection_command_input
        self.lifecycle.execute(now)
        self.events.append(
          dict(
            time=now,
            event="motion_selected",
            **swing.event,
            deadline=self.planner.plan[1],
            target_source=self.target_source,
            raw_command=self.raw_command.diagnostic if self.raw_command else None,
          )
        )
    target = self.executor.act(state, now)
    self.command_policy = self.executor.policy_name
    self.command_debug = self.executor.debug
    self.command_debug.update(
      target_source=self.target_source,
      planner_target_mode=self.target_mode if self.planner_active else None,
      raw_racket_command=self.raw_command.diagnostic if self.raw_command else None,
      return_frame_index=self.return_frames_sent
      if self.lifecycle.state == "RECOVER"
      else None,
    )
    self.debug.update(self.command_debug)
    self.previous_target = target.copy()
    if self.lifecycle.state == "RECOVER":
      self.return_frames_sent += 1
    return target

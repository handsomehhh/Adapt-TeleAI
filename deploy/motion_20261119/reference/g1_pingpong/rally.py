"""One Python controller for home stance, ball-based selection and rally recovery.

No MuJoCo, DDS, ROS or ball-generator truth enters the decision path. All motion
selection and hit inference use the measured robot state and ball positions.
"""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import Controller, rotation
from deploy.motion_20261119.reference.g1_pingpong.lifecycle import Lifecycle
from deploy.motion_20261119.reference.g1_pingpong.motions import MotionLibrary, RebasedMotion, ReturnReference
from deploy.motion_20261119.reference.g1_pingpong.perception import BallEpisodeMonitor, ReceivePlanner
from deploy.motion_20261119.reference.g1_pingpong.standing import StandingController


class RallyController:
  def __init__(
    self,
    deployment,
    runtime,
    options,
    *,
    home_index=5,
    allow=None,
    selection="auto",
    receiving=True,
    auto_start=True,
    auto_arm=True,
  ):
    self.cfg = deployment.settings
    self.physical_limit_observer = None
    self.runtime, self.options = runtime, options
    if options.get("stage2_tail_mode", "full") not in ("full", "observed_terminal"):
      raise ValueError("Unknown stage2_tail_mode")
    configured_allow = options.get("motion_allow") if allow is None else allow
    self.library = MotionLibrary(
      deployment, [home_index] if selection == "fixed" else configured_allow
    )
    if not 0 <= home_index < len(self.library.motions):
      raise ValueError("Invalid home motion index")
    self.home_index, self.selection, self.receiving = home_index, selection, receiving
    self.auto_start, self.auto_arm = auto_start, auto_arm
    self.home = self.library.motions[home_index]
    self.home_frozen = False
    self.manual_policy_entry = False
    self.lifecycle = Lifecycle(self.home.q[0], runtime["transition_seconds"])
    self.standing = StandingController(self.cfg, self.home, deployment.tracker, runtime)
    self.controller = Controller(
      self.cfg, self.home, deployment.tracker, deployment.actor
    )
    self.planner = None
    self.flight_monitor = None
    self.motion_index = home_index
    self.completed = self.recovered = 0
    self.ready_since = None
    self.events, self.debug = [], {}
    self.previous_target = None
    self.return_reference = None
    self.last_swing_estimate = None

  @property
  def candidate_indices(self):
    return self.library.allow if self.selection == "auto" else [self.home_index]

  # Runtime-facing view shared with deploy.motion_20261119.reference.hope_g1.controller.HopeController.
  @property
  def home_pose(self):
    return self.library.motions[self.home_index].q[0]

  @property
  def motion_name(self):
    return self.library.paths[self.motion_index].name

  @property
  def motion_files(self):
    return [
      self.library.paths[i]
      for i in sorted(set([self.home_index, *self.candidate_indices]))
    ]

  def describe(self):
    return (
      f"[rally] home/prepare/return={self.home_index} "
      f"({self.library.paths[self.home_index].name}); "
      f"swing_select={self.selection}; swing_candidates={self.candidate_indices}"
    )

  def candidates(self, state):
    return [
      (index, self.library.rebase(index, state)) for index in self.candidate_indices
    ]

  @property
  def phase(self):
    return {
      "WAIT": "WAIT_STATE",
      "PREPARE": "PREPARE",
      "POSITIONING": "POSITIONING",
      "POSE_HOLD": "POSE_HOLD",
      "TO_STABLE": "TO_STABLE",
      "STABLE": "STABLE",
      "POLICY_POSTURE": "POLICY_POSTURE",
      "READY": "IDLE",
      "ARMED": "IDLE",
      "EXECUTE": "SWING",
      "RECOVER": "RETURN",
      "HALT": "HALT",
    }[self.lifecycle.state]

  def select(self, state, estimate, now):
    choices = []
    timed_candidates = 0
    for index, reference in self.candidates(state):
      planner = ReceivePlanner(
        reference,
        self.cfg["ball"],
        spatial_tolerance=[np.inf] * 3,
        timing_tolerance=self.runtime["planner"]["timing_tolerance"],
        max_start_frame=self.options["max_start_frame"],
      )
      if not planner.try_start(estimate, now):
        self.debug = {"decision": {**planner.diagnostic, "motion_index": index}}
        continue
      assert planner.plan is not None
      timed_candidates += 1
      distance = float(
        np.linalg.norm(planner.plan[0] - reference.strike_position(self.cfg["ball"]))
      )
      if distance <= self.options["max_intercept_distance"]:
        choices.append((distance, index, reference, planner))
    if not choices:
      if timed_candidates:
        self.debug = {
          "decision": {
            "reason": "intercept_unreachable",
            "max_distance": self.options["max_intercept_distance"],
          }
        }
      return None
    best = min(choices, key=lambda item: (item[0], item[1]))
    current = next(
      (choice for choice in choices if choice[1] == self.motion_index), None
    )
    if (
      current is not None
      and current[0] - best[0] < self.options["selection_hysteresis"]
    ):
      return current
    return best

  def reset_manual_policy(self, now):
    """Discard a partially played motion before returning to the stable policy."""
    self.events.append(
      {"time": now, "event": "returned_to_stable", "from": self.lifecycle.state}
    )
    self.home = self.library.motions[self.home_index]
    self.home_frozen = False
    self.standing = StandingController(
      self.cfg, self.home, self.standing.controller.tracker, self.runtime
    )
    self.controller.reset()
    self.controller.motion = self.home
    self.planner = self.return_reference = self.previous_target = self.ready_since = (
      None
    )
    self.flight_monitor = None
    self.motion_index = self.home_index
    self.last_swing_estimate = None
    self.debug = {}
    self.lifecycle.state = "WAIT"

  def start_policy(self, target, now, *, reset_history=False, blend_reference=True):
    """Explicit A handoff after PD positioning; arm reception immediately."""
    if self.lifecycle.state not in ("POSE_HOLD", "STABLE"):
      raise RuntimeError("Policy entry requires POSE_HOLD")
    self.entry_started = now
    self.entry_reference = None
    self.blend_entry_reference = blend_reference
    self.manual_policy_entry = True
    self.auto_arm = True
    self.previous_target = target.copy()
    self.standing.previous = target.copy()
    controller = self.standing.controller
    if reset_history:
      controller.reset()
    else:
      controller.previous_joint = (
        (target - controller.default) / controller.scale
      ).astype(np.float32)
    self.lifecycle.state = "ARMED"
    self.events.append({"time": now, "event": "policy_enabled"})

  def step(self, state, estimate, now):
    if self.lifecycle.state == "HALT":
      return None
    if self.lifecycle.state == "WAIT":
      if not self.auto_start:
        return None
      self.lifecycle.prepare(state.q, now)
    if not self.home_frozen and self.lifecycle.state in ("PREPARE", "ARMED"):
      # HOPE freezes the home clip's first odometry-based train->world rebase
      # at policy entry. Raw motion-file coordinates are not the live home.
      # Capture once: subsequent recoveries must never move this world target.
      self.home = RebasedMotion(self.home, state.anchor_pos, state.anchor_quat)
      self.standing.motion = self.home
      self.standing.controller.motion = self.home
      self.controller.motion = self.home
      self.home_frozen = True
      self.events.append(
        {
          "time": now,
          "event": "home_frozen",
          "home": self.home_index,
          "anchor_position": state.anchor_pos.tolist(),
          "translation": self.home.translation.tolist(),
          "quaternion_wxyz": self.home.quaternion.tolist(),
        }
      )
    entry_seconds = (
      self.runtime.get("stable_policy", {}).get("transition_seconds", 0)
      if self.manual_policy_entry and getattr(self, "blend_entry_reference", True)
      else 0
    )
    if entry_seconds and now - self.entry_started <= entry_seconds:
      if self.entry_reference is None:
        self.entry_reference = ReturnReference(state, self.home)
      self.entry_reference.alpha = float(
        np.clip((now - self.entry_started) / entry_seconds, 0, 1)
      )
      self.standing.controller.motion = self.entry_reference
    elif (
      self.manual_policy_entry and getattr(self, "entry_reference", None) is not None
    ):
      self.standing.controller.motion = self.home
      self.entry_reference = None
    tilt = float(np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1))))
    if tilt > self.runtime["max_tilt_degrees"]:
      error = ValueError("Robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("rally_tilt", error)
    before = self.lifecycle.state
    self.debug = {
      "standing": self.standing.debug,
      "decision": {
        "reason": "not_idle" if before in ("EXECUTE", "RECOVER") else "waiting_for_ball"
      },
    }
    if before in ("RECOVER", "READY", "ARMED") and self.return_reference is not None:
      self.return_reference.alpha = float(
        np.clip(
          (now - self.lifecycle.transition_start)
          / self.options["return_blend_seconds"],
          0,
          1,
        )
      )
      if self.return_reference.alpha >= 1:
        self.return_reference = None
        self.standing.controller.motion = self.home
    # Explicit manual entry grants reception immediately. Static readiness
    # thresholds are still used during recovery, not to revoke manual arming.
    if self.manual_policy_entry and self.lifecycle.state == "ARMED":
      stable, metrics = self.standing.observe(state)
      self.standing.phase = "ARMED"
      self.standing.debug = {
        **metrics,
        "stable": stable,
        "manual_armed": True,
        "readiness_mode": self.standing.readiness_mode,
      }
    else:
      self.standing.update_readiness(state, self.lifecycle, now, self.previous_target)
    if self.lifecycle.state == "READY":
      if before == "RECOVER":
        self.recovered += 1
        self.motion_index = self.home_index
        if self.return_reference is None:
          self.standing.controller.motion = self.home
        self.events.append({"time": now, "event": "recovered", "home": self.home_index})
      if self.ready_since is None:
        self.ready_since = now
      if (
        self.receiving
        and self.auto_arm
        and now - self.ready_since >= self.options["ready_seconds"]
      ):
        self.lifecycle.arm(now, True)
    else:
      self.ready_since = None
    if self.receiving and self.lifecycle.state == "ARMED" and estimate is not None:
      choice = self.select(state, estimate, now)
      if choice is not None:
        distance, index, reference, self.planner = choice
        self.motion_index = index
        self.flight_monitor = BallEpisodeMonitor(self.cfg["ball"])
        self.return_reference = None
        self.standing.controller.motion = self.home
        self.controller.motion = reference
        self.controller.start_from(self.standing.controller)
        start_frame = int(self.planner.diagnostic["start_frame"])
        self.controller.t = start_frame / reference.fps
        entry_q, entry_anchor, _ = reference.sample(self.controller.t)
        self.lifecycle.execute(now)
        self.events.append(
          {
            "time": now,
            "event": "motion_selected",
            "motion_index": index,
            "motion": self.library.paths[index].name,
            "hit_type": self.library.hit_types[index],
            "distance": distance,
            "remaining": self.planner.plan[1] - now,
            "start_frame": start_frame,
            "entry_joint_error_max": float(np.max(np.abs(entry_q - state.q))),
            "entry_anchor_error_m": float(
              np.linalg.norm(entry_anchor - state.anchor_pos)
            ),
          }
        )
    if self.lifecycle.state == "EXECUTE":
      assert self.planner is not None
      ball_extrapolated = False
      terminal_mode = (
        self.options.get("stage2_tail_mode", "full") == "observed_terminal"
      )
      if estimate is not None:
        self.last_swing_estimate = estimate
      elif self.runtime.get("continue_swing_without_ball", False):
        # HOPE latches the receive goal through the swing. This Stage2 policy
        # additionally needs ball state: extrapolate the last measured flight,
        # retaining its original timestamp rather than inventing a fresh sample.
        estimate = self.last_swing_estimate
        ball_extrapolated = estimate is not None
      if estimate is None and not terminal_mode:
        raise TimeoutError("Ball positions expired during swing")
      ball = (
        self.planner.observation(estimate, now, state.racket_pos)
        if estimate is not None
        else None
      )
      terminal_before = self.flight_monitor.reason if self.flight_monitor else None
      observed_reason = (
        self.flight_monitor.update(ball, now) if self.flight_monitor else None
      )
      terminal_reason = observed_reason if terminal_mode else None
      if observed_reason is not None and terminal_before is None:
        self.events.append(
          dict(
            time=now,
            event="stage2_training_boundary_observed",
            reason=observed_reason,
            stage2_continues=not terminal_mode,
            motion_index=self.motion_index,
            reference_time=self.controller.t,
          )
        )
      # Continue the reference clip with Stage1 after the observed boundary of
      # the trained Stage2 episode. This does not wait, reset, or stop physics.
      follow = self.controller.t >= self.controller.motion.strike_time(self.cfg["ball"])
      target = self.controller.step(
        state, ball, tracker_only=terminal_reason is not None or ball_extrapolated
      )
      self.command_policy = (
        "tracker_ball_dropout"
        if ball_extrapolated
        else ("tracker_tail" if terminal_reason is not None else "hierarchical")
      )
      self.command_debug = self.controller.debug
      self.debug = {
        "ball_extrapolated": ball_extrapolated,
        "policy_phase": "tracker_ball_dropout"
        if ball_extrapolated
        else (
          "tracker_follow_through"
          if terminal_reason is not None
          else ("follow_through" if follow else "receive")
        ),
        "stage2_terminal_reason": terminal_reason,
        "observed_training_boundary": observed_reason,
        **self.controller.debug,
      }
      if self.controller.t >= self.controller.motion.duration:
        self.completed += 1
        self.lifecycle.recover(state.q, now)
        self.standing.resume_from(self.controller)
        if self.options["return_blend_seconds"] > 0:
          self.return_reference = ReturnReference(state, self.home)
          self.standing.controller.motion = self.return_reference
        self.events.append(
          {"time": now, "event": "swing_finished", "motion_index": self.motion_index}
        )
    else:
      target = self.standing.command(state)
      self.command_policy = "tracker"
      self.command_debug = self.standing.controller.debug
      self.debug["standing"] = self.standing.debug
    if target is None:
      raise RuntimeError("No control target in active rally phase")
    self.previous_target = target.copy()
    return target

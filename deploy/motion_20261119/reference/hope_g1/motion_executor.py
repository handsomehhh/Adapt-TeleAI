"""Reference-tracking executor: one HOPE actor, its motions and motion selector.

Goals latch at selection unless live command updates are enabled; optional live
reselection also rematches the motion. Idle/return use frame zero of a fixed home clip,
with an optional base XY latched at swing completion, and the same actor;
the motion clock advances only while swinging. No tracker is
substituted. See deploy.motion_20261119.reference.hope_g1.executor for the contract.
"""

from dataclasses import dataclass, replace

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import RACKET_HANDS, quaternion_multiply, rotation
from deploy.motion_20261119.reference.g1_pingpong.motions import MotionLibrary, RebasedMotion
from deploy.motion_20261119.reference.g1_pingpong.perception import ReceivePlanner
from deploy.motion_20261119.reference.hope_g1.a3_planner import ReceivePlan, command_goal
from deploy.motion_20261119.reference.hope_g1.clip_table import ClipTable
from deploy.motion_20261119.reference.hope_g1.configuration import TASK
from deploy.motion_20261119.reference.hope_g1.executor import Swing
from deploy.motion_20261119.reference.hope_g1.motion_selectors import build_motion_selector, check_command_interface
from deploy.motion_20261119.reference.racket_command import RacketCommandArray
from deploy.motion_20261119.reference.hope_g1.base_target_compensator import BaseTargetCompensator
from deploy.motion_20261119.reference.hope_g1.base_target_transition import build_base_target_transition


class AlignedLibrary:
  """Clips in the frozen home world, built on first use and then reused."""

  def __init__(self, motions, home):
    self.motions, self.home, self.cache = motions, home, {}

  def __len__(self):
    return len(self.motions)

  def __getitem__(self, index):
    if index not in self.cache:
      _, p, q = self.motions[index].sample(0)
      self.cache[index] = RebasedMotion(
        self.motions[index],
        self.home.rotation @ p + self.home.translation,
        quaternion_multiply(self.home.quaternion, q),
      )
    return self.cache[index]


@dataclass
class RacketGoal:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  base_xy: np.ndarray


HAND_SIGN = {"right": 1.0, "left": -1.0}
IDLE_HOME_MODES = ("last_hand", "fixed")


def mirror_motion_stem(stem):
  """HOPE scripts/mirror_motion_folder.py naming: <stem> <-> <stem>--mirror_left."""
  suffix = "--mirror_left"
  return stem[: -len(suffix)] if stem.endswith(suffix) else stem + suffix


class HopeActorController:
  """Construct the saved 175-D (dual-hand: 176-D) actor observation in Isaac order."""

  def __init__(self, cfg, policy):
    self.cfg, self.policy = cfg, policy
    # Dual-racket HOPE appends racket_hand_sign of the reference clip.
    self.dual_hand = bool(cfg.get("hope", {}).get("dual_hand", False))
    self.input_dim = 176 if self.dual_hand else 175
    if getattr(policy, "input_dim", self.input_dim) != self.input_dim:
      raise ValueError(f"HOPE actor expects {policy.input_dim} inputs, observation has {self.input_dim}")
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

  def observation(self, state, reference, frame, goal, time_to_strike, now, racket_hand="right"):
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
    if self.dual_hand:
      terms.append([HAND_SIGN[racket_hand]])
    elif racket_hand != "right":
      raise ValueError("A single-racket HOPE actor cannot swing a left-hand clip")
    observation = np.concatenate(terms).astype(np.float32)
    if observation.shape != (self.input_dim,) or not np.isfinite(observation).all():
      raise ValueError("Invalid HOPE observation")
    return observation, velocity

  def step(self, state, reference, frame, goal, time_to_strike, now, racket_hand="right"):
    obs, velocity = self.observation(
      state, reference, frame, goal, time_to_strike, now, racket_hand
    )
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
      racket_hand=racket_hand,
    )
    return target


class MotionExecutor:
  """HOPE actor + motion library + motion selector as one executor."""

  policy_name = "hope_actor"

  def __init__(
    self,
    deployment,
    runtime,
    options,
    *,
    command_driven,
    home_index=0,
    allow=None,
    selection="fixed",
  ):
    self.cfg, self.runtime, self.options = deployment.settings, runtime, options
    self.command_driven = command_driven
    self.a3_options = options.get("a3", {})
    self.target_mode = self.a3_options.get("target_mode", "raw")
    self.reference_mode = self.a3_options.get("reference_mode", "a3_live")
    self.live_motion_reselection = self.a3_options.get("live_motion_reselection", False)
    if not isinstance(self.live_motion_reselection, bool):
      raise ValueError("a3.live_motion_reselection must be a boolean")
    if self.live_motion_reselection and (
      not self.a3_options.get("live_command_updates", False)
      or not command_driven or self.target_mode != "raw"
    ):
      raise ValueError("a3.live_motion_reselection requires live_command_updates and raw planner commands")
    if self.target_mode not in ("raw", "envelope") or self.reference_mode not in (
      "a3_live",
      "training_common",
    ):
      raise ValueError("Invalid A3 target_mode/reference_mode")
    if self.cfg["task"] != TASK:
      raise ValueError("MotionExecutor requires the independent HOPE task")
    if selection not in ("fixed", "auto"):
      raise ValueError("HOPE motion selection must be fixed or auto")
    if (
      not np.isfinite(options["max_intercept_distance"])
      or options["max_intercept_distance"] <= 0
      or not 0 <= options["max_start_frame"] <= self.cfg["hope"]["max_start_frame"]
    ):
      raise ValueError("Invalid HOPE interception distance/start-frame window")
    self.selection = selection
    self.post_swing_base_target = options.get("post_swing_base_target", "home")
    if self.post_swing_base_target not in ("home", "current"):
      raise ValueError("post_swing_base_target must be home or current")
    self.post_strike_tracking_frames = options.get("post_strike_tracking_frames")
    if self.post_strike_tracking_frames is not None and (
      isinstance(self.post_strike_tracking_frames, bool)
      or not isinstance(self.post_strike_tracking_frames, int)
      or self.post_strike_tracking_frames < 0
    ):
      raise ValueError("post_strike_tracking_frames must be null or a nonnegative integer")
    self.motion_selector = build_motion_selector(options)
    self.base_compensator = BaseTargetCompensator(options.get("base_target_compensation"))
    self.base_transition = build_base_target_transition(options.get("base_target_transition"))
    if self.motion_selector.mode == "height_band" and (not command_driven or self.target_mode != "raw"):
      raise ValueError("height_band selector requires raw planner commands")
    self.library = MotionLibrary(
      deployment, [home_index] if selection == "fixed" else allow
    )
    self.library.hit_types = self.cfg["hope"]["hit_types"]
    if not 0 <= home_index < len(self.library.motions):
      raise ValueError("Invalid HOPE home index")
    self.home_index = self.motion_index = self.initial_home_index = home_index
    # Strike frames come only from configuration, resolved as Isaac does for
    # this checkpoint. Motion's own NPZ annotation is not zero-based.
    strike_frames = self.cfg["hope"]["strike_frames"]
    # Each clip's racket hand (dual-racket HOPE); strike points, goals and the
    # racket_hand observation all follow it. Single-racket tasks: all right.
    self.racket_hands = list(
      self.cfg["hope"].get("racket_hands") or ["right"] * len(self.library.motions)
    )
    if len(self.racket_hands) != len(self.library.motions) or any(
      h not in RACKET_HANDS for h in self.racket_hands
    ):
      raise ValueError("One right/left racket hand per HOPE motion is required")
    for clip, strike, hand in zip(
      self.library.motions, strike_frames, self.racket_hands, strict=True
    ):
      clip.annotated_strike = strike / clip.fps
      clip.racket_hand = hand
    self.hand_homes = self.idle_home_indices(options.get("dual_hand"), home_index, selection)
    self.clips = ClipTable(
      self.library.motions,
      strike_frames,
      self.cfg["ball"],
      max_start_frame=options["max_start_frame"],
      min_lead_frames=self.cfg["hope"]["min_lead_frames"],
    )
    self.home = self.library.motions[home_index]
    # Every clip of a library starts from this posture (frame zero of home).
    self.home_pose = self.home.q[0]
    self.actor = HopeActorController(self.cfg, deployment.actor)
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
    self.decision = None
    self.reset()

  def idle_home_indices(self, config, home_index, selection):
    """Idle home clip per racket hand, or None to always idle on home_index.

    Isaac returns to frame zero of the clip just swung, whose stance is
    mirrored for left-hand clips. ``last_hand`` (dual-hand auto default) idles
    on the home clip or its mirror, matching the hand of the last swing.
    """
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config) - {"idle_home", "mirror_home_index"}:
      raise ValueError("dual_hand must be a mapping of idle_home/mirror_home_index")
    dual = bool(self.cfg["hope"].get("dual_hand", False))
    default = "last_hand" if dual and selection == "auto" else "fixed"
    mode = config.get("idle_home", default)
    if mode not in IDLE_HOME_MODES:
      raise ValueError(f"dual_hand.idle_home must be one of {IDLE_HOME_MODES}")
    self.idle_home_mode = mode
    if mode == "fixed":
      return None
    if not dual or selection != "auto":
      raise ValueError("dual_hand.idle_home=last_hand requires a dual-hand policy and auto selection")
    mirror = config.get("mirror_home_index")
    if mirror is None:
      stem = mirror_motion_stem(self.library.paths[home_index].stem)
      matches = [i for i, p in enumerate(self.library.paths) if p.stem == stem]
      if len(matches) != 1:
        raise ValueError(
          f"No mirrored home clip {stem}.npz; set dual_hand.mirror_home_index or idle_home: fixed"
        )
      mirror = matches[0]
    if isinstance(mirror, bool) or not isinstance(mirror, int) or not 0 <= mirror < len(self.racket_hands):
      raise ValueError("dual_hand.mirror_home_index must be a motion index")
    homes = {self.racket_hands[home_index]: home_index, self.racket_hands[mirror]: mirror}
    if len(homes) != 2:
      raise ValueError("The mirrored home clip must use the other racket hand")
    return homes

  def reset(self):
    self.actor.reset()
    self.base_compensator.reset()
    self.base_transition.reset()
    self.home_index = self.initial_home_index
    self.home = self.library.motions[self.home_index]
    self.home_frozen = False
    self.references = None
    self.reference = self.goal = self.home_goal = self.plan = None
    self.idle_goal = None
    self.frozen_homes = {}
    self.motion_index = self.home_index
    self.frame = 0
    self.tracking_end_frame = self.last_tracked_frame = None
    self.swinging = False

  @property
  def debug(self):
    return self.actor.debug

  @property
  def candidate_indices(self):
    return [self.home_index] if self.selection == "fixed" else self.library.allow

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
    candidates = self.candidate_indices
    if len(candidates) > 32:
      candidates = f"{len(candidates)} clips"
    return (
      f"home={self.home_index} ({self.library.paths[self.home_index].name}); "
      f"selection={self.selection}; selector={self.motion_selector.mode}; candidates={candidates}; "
      f"actor={self.actor.input_dim}->29; idle_home={self.idle_home_mode}"
      + (f" {self.hand_homes}" if self.hand_homes else "") + "; "
      f"base_compensation={self.base_compensator.mode}; "
      f"base_transition={self.base_transition.mode}; "
      f"post_swing_base_target={self.post_swing_base_target}; "
      f"live_motion_reselection={self.live_motion_reselection}; "
      f"post_strike_tracking_frames={self.post_strike_tracking_frames}"
    )

  def candidates(self, state):
    if self.command_driven and self.reference_mode == "a3_live":
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
    return AlignedLibrary(self.library.motions, home)

  def candidate_strikes(self, state, indices):
    """Strike points of the references choose would build."""
    if self.reference_mode == "a3_live":
      return self.clips.rebased_strikes(indices, state.anchor_pos, state.anchor_quat)
    home = (self.references or self.aligned_library(state)).home
    return self.clips.aligned_strikes(indices, home)

  def reference_goal(self, reference, index, *, home=False):
    cfg = self.cfg["hope"]["racket"]
    f = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    source = reference.source
    # Left clips: left wrist + Rx(pi) mount (DualHandRacketTargetCommand).
    left = self.racket_hands[index] == "left"
    body = source.body_names.index(cfg["left_wrist_body_name" if left else "wrist_body_name"])
    wrist_q = source.body_quat[f, body]
    mount_q = quaternion_multiply(wrist_q, cfg["left_mount_quat" if left else "mount_quat"])
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

  def prepare(self, state, now):
    """Freeze every clip and the idle goal in the world of the ready pose."""
    self.references = self.aligned_library(state)
    self.home = self.references[self.home_index]
    # A3 derives the idle/return base target from the home strike-frame reach,
    # the reference baseline from the frame-zero pelvis XY.
    self.home_goal = self.reference_goal(
      self.home, self.home_index, home=not self.command_driven
    )
    self.idle_goal = self.home_goal
    for hand, index in (self.hand_homes or {}).items():
      reference = self.references[index]
      self.frozen_homes[hand] = (
        index, reference, self.reference_goal(reference, index, home=not self.command_driven)
      )
    self.home_frozen = True
    return dict(
      translation=self.home.translation.tolist(),
      quaternion_wxyz=self.home.quaternion.tolist(),
    )

  def choose_from_ball(self, state, estimate, now):
    """Reference baseline: start the clip whose strike a ball estimate meets."""
    self.decision = None
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
        self.decision = {**planner.diagnostic, "motion_index": index}
        continue
      if not self.clips.startable(index, planner.diagnostic["start_frame"]):
        self.decision = dict(
          planner.diagnostic, reason="start_inside_min_lead", motion_index=index
        )
        continue
      distance = float(
        np.linalg.norm(planner.plan[0] - reference.strike_position(self.cfg["ball"]))
      )
      offset_world = planner.plan[0] - reference.strike_position(self.cfg["ball"])
      # The training perturbation box is axis-aligned in NPZ/Isaac world,
      # exactly like the absolute-vector terms in the actor observation.
      offset = reference.rotation.T @ offset_world
      if np.any(np.abs(offset) > self.target_offset_limit):
        self.decision = dict(
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

  def choose(self, state, command, now):
    return self.motion_selector.choose(self, state, command, now)

  def begin(self, choice, now):
    self.base_compensator.begin()
    return self._apply_choice(choice)

  def _apply_choice(self, choice):
    """Install a selected clip/goal without resetting actor or odometry history."""
    distance, self.motion_index, self.reference, self.plan = choice
    self.frame = int(self.plan.diagnostic["start_frame"])
    self.last_tracked_frame = None
    self.tracking_end_frame = len(self.reference.q) - 1
    if self.post_strike_tracking_frames is not None:
      self.tracking_end_frame = min(
        self.tracking_end_frame,
        int(self.clips.strike_frames[self.motion_index]) + self.post_strike_tracking_frames,
      )
    self.goal = self.reference_goal(self.reference, self.motion_index)
    # The explicit reference baseline only retargets position. The A3
    # branch below replaces it with the complete latched RacketCommand.
    displacement = self.plan.plan[0] - self.goal.position
    # RacketTargetCommand couples XY base placement to contact position:
    # target_base_xy = target_racket_xy - reference_reach_offset_xy.
    self.goal.base_xy = self.goal.base_xy + displacement[:2]
    self.goal.position = self.plan.plan[0].copy()
    command = None
    if self.command_driven:
      command = self.plan.command
      nominal = self.reference_goal(self.reference, self.motion_index)
      sign = (
        -1
        if self.library.hit_types[self.motion_index]
        in self.cfg["hope"]["racket"]["ref_velocity_backhand_values"]
        else 1
      )
      self.goal = RacketGoal(
        *command_goal(
          command,
          nominal,
          self.reference,
          self.cfg["hope"]["racket"],
          sign,
          self.target_mode,
          self.target_perturb_scale,
        )
      )
    self.swinging = True
    return Swing(
      plan=self.plan,
      command=command,
      event=dict(
        motion_index=self.motion_index,
        motion=self.motion_name,
        racket_hand=self.racket_hands[self.motion_index],
        distance=distance,
        start_frame=self.frame,
        selector_mode=self.motion_selector.mode,
        tracking_end_frame=self.tracking_end_frame,
        post_strike_tracking_frames=self.post_strike_tracking_frames,
        **{key: self.plan.diagnostic[key] for key in ("command_index", "center_index")
           if key in self.plan.diagnostic},
      ),
    )

  def swing_finished(self, now):
    # Each reference is held for a complete control interval. Finish on the
    # tick AFTER the inclusive tracking endpoint, including the strike frame.
    return self.swinging and self.frame > self.tracking_end_frame

  def update_command(self, state, command, now):
    """Retarget, optionally reselecting a clip, without resetting actor history.

    Initial selection retains the training start window. Refinements of an
    ongoing swing may arrive later, up to 40 ms before the requested strike.
    Align the current clip to the absolute deadline rather than restarting it.
    """
    check_command_interface(self.motion_selector, command)
    # The controller checks live updates before completion. Once an explicitly
    # capped swing has consumed its final frame, a fresh command must not rewind
    # it and bypass recovery. Preserve legacy behavior when no cap is configured.
    if self.post_strike_tracking_frames is not None and self.swing_finished(now):
      self.decision = dict(reason="live_command_tracking_complete",
                           tracking_end_frame=self.tracking_end_frame)
      return None
    if isinstance(command, RacketCommandArray):
      if not self.swinging or not self.command_driven or self.target_mode != "raw":
        self.decision = dict(reason="live_command_invalid")
        return None
      return self._reselect_command(state, command, now)
    remaining = command.deadline - now
    vectors = [np.asarray(v) for v in (command.position, command.velocity, command.normal)]
    if (not self.swinging or not self.command_driven or self.target_mode != "raw"
        or not np.isfinite(remaining) or any(v.shape != (3,) for v in vectors)
        or not all(np.isfinite(v).all() for v in vectors)
        or np.linalg.norm(vectors[2]) < 1e-6 or not command.diagnostic.get("valid", True)):
      self.decision = dict(reason="live_command_invalid")
      return None
    frame = int(self.clips.start_frames(np.array([self.motion_index]), remaining)[0])
    if (not .04 < remaining <= self.a3_options.get("max_command_time_to_strike", .5)
        or (frame < 0 and not self.live_motion_reselection)):
      self.decision = dict(reason="live_command_outside_time_window", remaining=remaining, frame=frame)
      return None
    if self.live_motion_reselection:
      return self._reselect_command(state, command, now)
    update_check = self.motion_selector.check_update(self, state, command, now)
    if update_check["reason"] != "accepted":
      self.decision = update_check
      return None
    nominal = self.reference_goal(self.reference, self.motion_index)
    sign = -1 if self.library.hit_types[self.motion_index] in self.cfg["hope"]["racket"]["ref_velocity_backhand_values"] else 1
    goal = RacketGoal(*command_goal(command, nominal, self.reference,
                                   self.cfg["hope"]["racket"], sign,
                                   self.target_mode, self.target_perturb_scale))
    old_frame, old_deadline = self.frame, self.plan.command.deadline
    self.decision = dict(update_check)
    self.decision.update(reason="live_command_accepted", motion_index=self.motion_index,
                         remaining=remaining, reference_frame=frame)
    self.goal, self.frame = goal, frame
    self.plan = ReceivePlan(command, dict(self.decision))
    return Swing(self.plan, command, dict(motion_index=self.motion_index, motion=self.motion_name,
                 previous_frame=old_frame, reference_frame=frame, previous_deadline=old_deadline,
                 selector_mode=self.motion_selector.mode))

  def _reselect_command(self, state, command, now):
    # Selection is read-only until a complete feasible choice is available.
    kwargs = {"fixed_motion": not self.live_motion_reselection} if isinstance(command, RacketCommandArray) else {}
    choice = self.motion_selector.choose(self, state, command, now, live_update=True, **kwargs)
    if choice is None:
      return None
    old_index, old_frame = self.motion_index, self.frame
    old_deadline = self.plan.command.deadline
    distance, index, reference, plan = choice
    self.decision = dict(self.decision, reason="live_command_accepted",
                         reference_frame=int(plan.diagnostic["start_frame"]))
    swing = self._apply_choice((distance, index, reference,
                                ReceivePlan(plan.command, dict(self.decision))))
    swing.event.update(
      live_motion_reselection=self.live_motion_reselection, motion_changed=index != old_index,
      previous_motion_index=old_index, previous_frame=old_frame,
      reference_frame=self.frame, previous_deadline=old_deadline,
    )
    return swing

  def finish(self, state):
    # Capture once at the EXECUTE -> RECOVER boundary, not on every idle tick.
    # Preserve the original home and all non-base targets for future swings.
    if self.frozen_homes:
      # Dual hand: return to the home stance of the hand that just swung.
      self.home_index, self.home, self.home_goal = self.frozen_homes[
        self.racket_hands[self.motion_index]
      ]
    self.idle_goal = self.home_goal
    if self.post_swing_base_target == "current":
      base_xy = np.asarray(state.root_pos, dtype=float)[:2].copy()
      if base_xy.shape != (2,) or not np.isfinite(base_xy).all():
        raise ValueError("Cannot latch nonfinite post-swing base XY")
      self.idle_goal = replace(self.home_goal, base_xy=base_xy)
    self.swinging = False
    motion_end = len(self.reference.q) - 1
    return dict(
      motion_index=self.motion_index, final_frame=self.last_tracked_frame,
      motion_final_frame=motion_end, tracking_end_frame=self.tracking_end_frame,
      post_strike_tracking_frames=self.post_strike_tracking_frames,
      post_swing_base_target=self.post_swing_base_target,
      racket_hand=self.racket_hands[self.motion_index],
      idle_home_index=self.home_index,
      idle_base_xy=self.idle_goal.base_xy.tolist(),
      finish_reason="post_strike_tracking_limit" if self.tracking_end_frame < motion_end else "motion_end",
    )

  def release(self):
    self.motion_index = self.home_index

  def act(self, state, now):
    # Return/idle keep the home reference; only idle base XY can be latched
    # at swing completion instead of using the original home goal.
    reference = self.reference if self.swinging else self.home
    frame = self.frame if self.swinging else 0
    goal = self.goal if self.swinging else self.idle_goal
    strike_frame = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    time_to_strike = (
      self.plan.command.deadline - now
      if self.swinging and self.a3_options.get("live_command_updates", False)
      else (strike_frame - frame) * self.cfg["control_dt"]
    )
    # The transition shapes the base target the actor sees while the physical
    # goal switches (idle <-> swing); compensation then acts on that target.
    transition_xy, transition_debug = self.base_transition.update(
      state, goal.base_xy, now=now, swinging=self.swinging, time_to_strike=time_to_strike,
    )
    base_xy, compensation_debug = self.base_compensator.update(
      state, transition_xy, now=now, time_to_strike=time_to_strike, swinging=self.swinging,
    )
    if self.base_compensator.mode != "active":
      base_xy = transition_xy
    actor_goal = goal if np.array_equal(base_xy, goal.base_xy) else replace(goal, base_xy=base_xy)
    target = self.actor.step(
      state,
      reference,
      frame,
      actor_goal,
      time_to_strike,
      now,
      self.racket_hands[self.motion_index if self.swinging else self.home_index],
    )
    self.actor.debug["reference_frame_count"] = len(reference.q)
    self.actor.debug["tracking_end_frame"] = self.tracking_end_frame if self.swinging else None
    self.actor.debug["true_goal_base_xy"] = goal.base_xy.copy()
    self.actor.debug["post_swing_base_target"] = self.post_swing_base_target
    self.actor.debug["base_target_compensation"] = compensation_debug
    self.actor.debug["base_target_transition"] = transition_debug
    if self.swinging:
      self.last_tracked_frame = min(max(int(frame), 0), len(reference.q) - 1)
      self.frame += 1
    return target

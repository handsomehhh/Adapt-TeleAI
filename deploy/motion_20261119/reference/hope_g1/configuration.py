"""Read HOPE's saved training contract without importing Isaac Sim."""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from deploy.motion_20261119.reference.g1_pingpong.configuration import Deployment
from deploy.motion_20261119.reference.g1_pingpong.core import RACKET_HANDS, mirror_name
from deploy.motion_20261119.reference.g1_pingpong.pd_gains import deployment_gains
from deploy.motion_20261119.reference.g1_pingpong.policy import training_config
from deploy.motion_20261119.reference.hope_g1.policy import HopePolicy

TASK = "HOPEPingPongMultiMotionG1"
# Planner-command training shares the same deployed actor and motion contract.
# The dual-racket variants (either hand swings) append one racket_hand input.
TRAINING_EXPERIMENTS = (
  "g1_hope_multi_motion",
  "g1_hope_planner_cmd",
  "g1_hope_multi_motion_dual_hand",
  "g1_hope_planner_cmd_dual_hand",
)
OBSERVATIONS = {
  "command": 58,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "joint_pos": 29,
  "joint_vel": 29,
  "actions": 29,
  "projected_gravity": 3,
  "base_target_pos_b": 2,
  "racket_target_pos_b": 3,
  "racket_target_vel_w": 3,
  "racket_target_normal_w": 3,
  "time_to_strike": 1,
}

# HOPE dual_hand.racket_hand_sign: +1 right-racket clip, -1 left (appended last).
DUAL_HAND_OBSERVATIONS = {**OBSERVATIONS, "racket_hand": 1}

_OBS_MODULE = "isaaclab.envs.mdp.observations:"
_MOTION_MODULE = "whole_body_tracking.tasks.tracking.mdp.observations:"
_HOPE_MODULE = "whole_body_tracking.tasks.tracking.mdp.hope_observations:"
OBSERVATION_FUNCTIONS = {
  **{n: _OBS_MODULE + n for n in ("base_lin_vel", "base_ang_vel", "projected_gravity")},
  "command": _OBS_MODULE + "generated_commands",
  "joint_pos": _OBS_MODULE + "joint_pos_rel",
  "joint_vel": _OBS_MODULE + "joint_vel_rel",
  "actions": _OBS_MODULE + "last_action",
  **{n: _MOTION_MODULE + n for n in ("motion_anchor_pos_b", "motion_anchor_ori_b")},
  **{
    n: _HOPE_MODULE + n
    for n in (
      "base_target_pos_b",
      "racket_target_pos_b",
      "racket_target_vel_w",
      "racket_target_normal_w",
      "time_to_strike",
    )
  },
  "racket_hand": "whole_body_tracking.tasks.tracking.mdp.dual_hand:racket_hand_sign",
}


def resolve_patterns(value, names, *, default=None):
  """Resolve Isaac regex dictionaries by joint name, rejecting ambiguity."""
  if not isinstance(value, dict):
    result = np.full(len(names), value, dtype=float)
  else:
    result = []
    for name in names:
      matches = [v for expr, v in value.items() if re.fullmatch(expr, name)]
      if len(matches) > 1 or (not matches and default is None):
        raise ValueError(f"Ambiguous/missing joint setting: {name}")
      result.append(matches[0] if matches else default)
    result = np.asarray(result, dtype=float)
  if result.shape != (len(names),) or not np.isfinite(result).all():
    raise ValueError("Invalid per-joint values")
  return result


# MultiMotionLoader(manifest="npz") reads the stroke from the first present key.
NPZ_STROKE_KEYS = ("hit_type", "target_face")
NPZ_STROKES = {
  "forward": "forehand",
  "forehand": "forehand",
  "backward": "backhand",
  "backhand": "backhand",
}


def motion_paths(motion, motion_dir=None, *, subset_from_dir=False):
  """Resolve an optional directory subset in saved training order."""
  paths = [Path(p) for p in motion["motion_files"]]
  if not paths or len({p.name for p in paths}) != len(paths):
    raise ValueError("Empty/duplicate HOPE motion library")
  indices = list(range(len(paths)))
  if motion_dir is not None:
    directory = Path(motion_dir)
    paths = [directory / p.name for p in paths]
  if subset_from_dir:
    if motion_dir is None:
      raise ValueError("motion_subset_from_dir requires motion_dir")
    if not directory.is_dir():
      raise FileNotFoundError(f"Motion directory does not exist: {directory}")
    available = {p.name for p in directory.glob("*.npz") if p.is_file()}
    unknown = available - {p.name for p in paths}
    if unknown:
      raise ValueError(f"Motion subset contains untrained files: {sorted(unknown)}")
    indices = [i for i, p in enumerate(paths) if p.name in available]
    paths = [paths[i] for i in indices]
    if not paths:
      raise ValueError(f"Empty HOPE motion subset: {directory}")
  return [p.resolve() for p in paths], indices


def motion_metadata(paths, motion, racket, *, motion_dir=None, motion_indices=None):
  """Per-clip zero-based strike frames and hit types, resolved as in Isaac.

  Without a manifest the strike frame is ``round(strike_phase * (frames - 1))``.
  With ``motion_manifest: npz`` it is the clip's ONE-based ``hope_strike_frame``
  minus one. Hit types try the JSON map first, then the NPZ stroke, then the
  configured default, like ``RacketTargetCommand._hit_type_for_motion``.
  """
  manifest = motion.get("motion_manifest")
  if manifest not in (None, "npz"):
    raise ValueError(f"Unsupported HOPE motion manifest: {manifest!r}")
  if manifest is None and motion.get("random_start_frame_min_lead_frames") is not None:
    raise ValueError("random_start_frame_min_lead_frames requires an NPZ manifest")
  hit_file = str(racket.get("ref_velocity_hit_type_file") or "").strip()
  hit_map = {}
  if hit_file:
    hit_path = Path(motion_dir) / Path(hit_file).name if motion_dir else Path(hit_file)
    hit_map = json.loads(hit_path.read_text())
  strike_frames, hit_types, rates = [], [], []
  indices = range(len(paths)) if motion_indices is None else motion_indices
  for index, path in zip(indices, paths, strict=True):
    with np.load(path, allow_pickle=False) as data:
      rates.append(float(np.asarray(data["fps"]).reshape(-1)[0]))
      frames = len(data["joint_pos"])
      stroke = next((str(data[k]) for k in NPZ_STROKE_KEYS if k in data), None)
      if manifest == "npz":
        if "hope_strike_frame" not in data:
          raise KeyError(f"Motion has no hope_strike_frame: {path}")
        strike = int(np.asarray(data["hope_strike_frame"]).reshape(-1)[0]) - 1
      else:
        strike = round(racket["strike_phase"] * (frames - 1))
    if not 0 <= strike < frames:
      raise ValueError(f"Strike frame outside motion: {path}")
    stroke = NPZ_STROKES.get((stroke or "").strip().lower())
    if manifest == "npz" and stroke is None:
      raise ValueError(f"Motion has no forehand/backhand stroke: {path}")
    keys = [str(motion["motion_files"][index]), path.name, path.stem, str(index)]
    mapped = next((hit_map[k] for k in keys if k in hit_map), None)
    strike_frames.append(strike)
    hit_types.append(
      mapped
      or (stroke if manifest == "npz" else None)
      or str(racket["ref_velocity_default_hit_type"]).strip().lower()
    )
  return strike_frames, hit_types, rates


NPZ_RACKET_HAND_KEY = "racket_hand"


def motion_racket_hands(paths, motion, *, motion_dir=None, motion_indices=None):
  """Per-clip racket hand, resolved as MultiMotionCommand._resolve_racket_hands.

  Priority: racket_hand_file entry > npz ``racket_hand`` > default_racket_hand
  (the dual-hand task saves None: every clip must be labelled).
  """
  hand_file = str(motion.get("racket_hand_file") or "").strip()
  hand_map = {}
  if hand_file:
    hand_path = Path(motion_dir) / Path(hand_file).name if motion_dir else Path(hand_file)
    hand_map = {str(k): str(v).strip().lower() for k, v in json.loads(hand_path.read_text()).items()}
  default = motion.get("default_racket_hand", "right")
  hands = []
  indices = range(len(paths)) if motion_indices is None else motion_indices
  for index, path in zip(indices, paths, strict=True):
    keys = [str(motion["motion_files"][index]), path.name, path.stem]
    hand = next((hand_map[k] for k in keys if k in hand_map), None)
    if hand is None:
      with np.load(path, allow_pickle=False) as data:
        if NPZ_RACKET_HAND_KEY in data:
          hand = str(data[NPZ_RACKET_HAND_KEY]).strip().lower()
    hand = hand or default
    if hand is None:
      raise ValueError(f"Motion has no racket_hand label and default_racket_hand is None: {path}")
    hand = str(hand).strip().lower()
    if hand not in RACKET_HANDS:
      raise ValueError(f"Motion {path} racket_hand={hand!r}; expected one of {RACKET_HANDS}")
    hands.append(hand)
  return hands


def validate_dual_racket(racket):
  """The deployed left racket is the mirror of the right, as HOPE trains it."""
  if (
    racket.get("left_wrist_body_name") != mirror_name(racket["wrist_body_name"])
    or racket.get("left_mount_offset") is not None
    or not np.allclose(racket.get("left_mount_quat", ()), [0.0, 1.0, 0.0, 0.0])
    or not np.allclose(racket["mount_quat"], [1.0, 0.0, 0.0, 0.0])
  ):
    raise ValueError(
      "Dual-hand deployment supports only the mirrored left racket: "
      "left_wrist=mirror(wrist), left_mount_offset=None, left_mount_quat=Rx(pi)"
    )


@dataclass
class HopeDeployment(Deployment):
  def report(self):
    strikes = self.settings["hope"]["strike_frames"]
    hits = self.settings["hope"]["hit_types"]
    motions = [dict(index=i, file=p.name) for i, p in enumerate(self.motions)]
    if len(motions) > 64:
      # Keep check output readable; the order is the saved env.yaml order.
      motions = [*motions[:3], "...", *motions[-3:]]
    return dict(
      task=TASK,
      controller="deploy.motion_20261119.reference.hope_g1.controller.HopeController",
      executor=self.settings["executor"],
      checkpoint=str(self.checkpoint),
      iteration=self.settings["iteration"],
      actor_input=self.actor.input_dim,
      dual_hand=self.settings["hope"]["dual_hand"],
      racket_hand_counts={
        h: self.settings["hope"]["racket_hands"].count(h)
        for h in RACKET_HANDS
        if h in self.settings["hope"]["racket_hands"]
      },
      actor_output=29,
      control_dt=self.settings["control_dt"],
      num_motions=len(self.motions),
      motions=motions,
      motion_manifest=self.settings["hope"]["motion"].get("motion_manifest"),
      strike_frame_range=[min(strikes), max(strikes)],
      min_lead_frames=self.settings["hope"]["min_lead_frames"],
      hit_type_counts={h: hits.count(h) for h in sorted(set(hits))},
      normalization="(obs - mean) / (std + 0.01)",
      self_collisions=self.settings["hope"]["self_collisions"],
      collision_source=self.settings["hope"]["collision_source"],
      world_frame=self.settings["world_frame"],
      simulation_world_translation=self.settings["simulation_world_translation"],
    )


def robot_contract(saved, run, names, action, ball_config, world_config):
  """Build the shared G1 plant and joint contract, without motion dependencies."""
  # Build this task's robot, floor and court without instantiating another task.
  from deploy.motion_20261119.reference.hope_g1.scene import table_cfg
  from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
    get_g1_pingpong_robot_cfg,
  )
  from mjlab.scene import Scene, SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  defaults = yaml.safe_load(
    (Path(__file__).parent / "configs/rally_mujoco.yaml").read_text()
  )
  ball_config = ball_config if ball_config is not None else defaults["ball"]
  world_config = world_config if world_config is not None else defaults["world"]
  from deploy.motion_20261119.reference.hope_g1.world import (
    mujoco_ball_config,
    simulation_translation,
    validate_table_world,
  )

  validate_table_world(ball_config)
  translation = simulation_translation(world_config)
  simulation_ball = mujoco_ball_config(ball_config, translation)

  robot = get_g1_pingpong_robot_cfg()
  # This task trained with its fast URDF and self-collisions enabled. Do not
  # silently inherit the other deploy task's foot spheres/disabled self contact.
  from functools import partial

  from deploy.motion_20261119.reference.hope_g1.assets import training_collision_spec

  asset_path = saved["scene"]["robot"]["spawn"].get("asset_path")
  if not isinstance(asset_path, str) or Path(asset_path).suffix.lower() != ".urdf":
    raise ValueError(
      f"Expected a training URDF in scene.robot.spawn.asset_path: {asset_path!r}"
    )
  robot_dir = run / "robot"
  urdf = robot_dir / Path(asset_path).name
  if not urdf.is_file():
    available = sorted(p.name for p in robot_dir.glob("*.urdf") if p.is_file())
    raise FileNotFoundError(
      f"No local URDF matches training asset_path {asset_path!r}: "
      f"expected {urdf}; available URDF files in {robot_dir}: {available}"
    )
  self_collisions = bool(
    saved["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"]
  )
  from deploy.motion_20261119.reference.hope_g1.assets import dual_racket_spec, urdf_has_left_racket

  dual_racket = urdf_has_left_racket(urdf)
  if dual_racket:
    # Left wrist mass/site from the dual-racket URDF, then its collisions.
    robot.spec_fn = partial(dual_racket_spec, robot.spec_fn, urdf)
  robot.spec_fn = partial(training_collision_spec, robot.spec_fn, urdf, self_collisions)
  robot.collisions = tuple(
    replace(
      c,
      geom_names_expr=("hope_training_shape_.*",),
      contype=1 if self_collisions else 0,
      conaffinity=1,
    )
    for c in robot.collisions
  )
  init = saved["scene"]["robot"]["init_state"]
  robot.init_state.joint_pos = init["joint_pos"]
  robot.init_state.joint_vel = init["joint_vel"]
  robot.init_state.pos = tuple(init["pos"])
  model = Scene(
    SceneCfg(
      entities={"robot": robot, "table": table_cfg(simulation_ball)},
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    device="cpu",
  ).compile()
  collision_ids = [
    i
    for i in range(model.ngeom)
    if model.geom(i).name.startswith("robot/hope_training_shape_")
  ]
  if (
    not collision_ids
    or not np.all(model.geom_conaffinity[collision_ids] == 1)
    or not np.all(model.geom_contype[collision_ids] == (1 if self_collisions else 0))
  ):
    raise ValueError(
      "Training collision masks were not propagated into the compiled scene"
    )
  joints = [model.joint("robot/" + n).id for n in names]
  default = resolve_patterns(init["joint_pos"], names, default=0.0)
  scales = resolve_patterns(action["scale"], names)
  kp, kd, torque, armature = [np.zeros(29) for _ in range(4)]
  covered = np.zeros(29, dtype=int)
  for actuator in saved["scene"]["robot"]["actuators"].values():
    indices = [
      i
      for i, n in enumerate(names)
      if any(re.fullmatch(p, n) for p in actuator["joint_names_expr"])
    ]
    subset = [names[i] for i in indices]
    for dest, key in (
      (kp, "stiffness"),
      (kd, "damping"),
      (torque, "effort_limit_sim"),
      (armature, "armature"),
    ):
      dest[indices] = resolve_patterns(actuator[key], subset)
    covered[indices] += 1
  if not np.all(covered == 1) or np.any(scales <= 0):
    raise ValueError("Invalid actuator coverage or action scales")
  # The shared runtime uses literal execution gains; require agreement up front.
  execution = deployment_gains(names)
  if not np.allclose(kp, execution[0]) or not np.allclose(kd, execution[1]):
    raise ValueError("HOPE training PD differs from the shared execution PD table")
  model.dof_armature[model.jnt_dofadr[joints]] = armature
  for j, limit in zip(joints, torque, strict=True):
    a = np.flatnonzero(model.actuator_trnid[:, 0] == j)
    if len(a) != 1:
      raise ValueError("Expected one position servo per G1 joint")
    model.actuator_forcerange[a[0]] = [-limit, limit]
  dt = float(saved["sim"]["dt"])
  control_dt = dt * saved["decimation"]
  if not np.isclose(control_dt, 0.02):
    raise ValueError("HOPE task must run at 50Hz")
  settings = dict(
    ball=dict(ball_config),
    world_frame="hope_table",
    simulation_world_translation=translation.tolist(),
    simulation_ball=simulation_ball,
    control_dt=control_dt,
    physics_dt=dt,
    decimation=saved["decimation"],
    sim_options=dict(integrator="implicitfast", timestep=dt, iterations=20),
    robot=dict(
      joint_names=names,
      default_q=default.tolist(),
      default_dq=resolve_patterns(init["joint_vel"], names).tolist(),
      kp=kp.tolist(),
      kd=kd.tolist(),
      torque_limits=torque.tolist(),
      joint_limits=model.jnt_range[joints].tolist(),
      action_scale=scales.tolist(),
    ),
  )
  return model, settings, urdf, self_collisions


def load_deployment(
  checkpoint,
  *,
  motion_dir=None,
  motion_subset_from_dir=False,
  env_config=None,
  agent_config=None,
  ball_config=None,
  world_config=None,
):
  checkpoint = Path(checkpoint).resolve()
  run = checkpoint.parent
  saved = training_config(env_config or run / "params/env.yaml")
  agent = training_config(agent_config or run / "params/agent.yaml")
  if agent.get("experiment_name") not in TRAINING_EXPERIMENTS:
    raise ValueError(
      f"Expected a HOPE training task in {TRAINING_EXPERIMENTS}, "
      f"got {agent.get('experiment_name')!r}"
    )
  group = saved["observations"]["policy"]
  terms = {k: v for k, v in group.items() if isinstance(v, dict) and "func" in v}
  dual_hand = list(terms) == list(DUAL_HAND_OBSERVATIONS)
  actor_terms = DUAL_HAND_OBSERVATIONS if dual_hand else OBSERVATIONS
  if list(terms) != list(actor_terms):
    raise ValueError(
      "HOPE actor observation order differs from the 175-D (or dual-hand 176-D) contract"
    )
  if dual_hand != agent["experiment_name"].endswith("_dual_hand"):
    raise ValueError("racket_hand observation and *_dual_hand experiment must agree")
  if not group["concatenate_terms"] or group.get("history_length"):
    raise ValueError("Unsupported HOPE observation history/layout")
  for name, term in terms.items():
    if term["func"] != OBSERVATION_FUNCTIONS[name]:
      raise ValueError(f"HOPE observation function differs: {name}")
    if any(term.get(k) is not None for k in ("scale", "clip", "modifiers")) or term.get(
      "history_length", 0
    ):
      raise ValueError(f"Unsupported HOPE observation transformation: {name}")
  motion = saved["commands"]["motion"]
  racket = saved["commands"]["racket_target"]
  names = list(motion["joint_names"])
  if len(names) != 29 or len(set(names)) != 29:
    raise ValueError("Expected all 29 G1 joints in saved Isaac articulation order")
  action = saved["actions"]["joint_pos"]
  if (
    not action["use_default_offset"]
    or action.get("clip") is not None
    or action["joint_names"] != [".*"]
    or action.get("preserve_order", False)
    or any(action.get("delay_steps_range", [0, 0]))
    or any(action.get("actuator_lag_time_constant_range", [0, 0]))
  ):
    raise ValueError("Unsupported HOPE joint action contract")
  if (
    racket["base_target_pos_dim"] != 2
    or racket["ref_velocity_source"] != "normal_fixed"
  ):
    raise ValueError("Expected XY base target and normal_fixed racket velocity")
  if (
    motion["anchor_body_name"] != "torso_link"
    or motion.get("motion_manifest") not in (None, "npz")
    or not motion["wrap_to_start_without_reset"]
    # early_stop_tracking_enabled is accepted: Isaac ends tracking at a random
    # frame in [strike + n, last); deployment caps it via the executor's
    # post_strike_tracking_frames option (rally config), not this contract.
    # Deployment idles on frame zero of a fixed home clip, as Isaac does on
    # the pending clip when return motions are not resampled.
    or motion.get("return_resample_motion_enabled", False)
  ):
    raise ValueError("Unsupported HOPE anchor, strike manifest or wrap contract")
  paths, motion_indices = motion_paths(
    motion, motion_dir, subset_from_dir=motion_subset_from_dir
  )
  model, common, urdf, self_collisions = robot_contract(
    saved, run, names, action, ball_config, world_config
  )
  if dual_hand:
    validate_dual_racket(racket)
    from deploy.motion_20261119.reference.hope_g1.assets import urdf_has_left_racket

    if not urdf_has_left_racket(urdf):
      raise ValueError(f"Dual-hand policy requires the dual-racket training URDF, got {urdf}")
    racket_hands = motion_racket_hands(
      paths, motion, motion_dir=motion_dir, motion_indices=motion_indices
    )
  else:
    racket_hands = ["right"] * len(paths)
  control_dt = common["control_dt"]
  # Clip arrays are validated once by the controller's MotionLibrary; here only
  # the per-clip timing metadata is read, which keeps 7k-clip libraries fast.
  strike_frames, hit_types, rates = motion_metadata(
    paths, motion, racket, motion_dir=motion_dir, motion_indices=motion_indices
  )
  # Keep the deployed contract aligned with the compact, selected library.
  motion = dict(motion, motion_files=[motion["motion_files"][i] for i in motion_indices])
  for path, rate in zip(paths, rates, strict=True):
    if not np.isclose(rate, 1 / control_dt):
      raise ValueError(f"Motion fps differs from policy rate: {path}")
  policy = HopePolicy(checkpoint, agent, input_dim=sum(actor_terms.values()))
  settings = dict(
    **common,
    task=TASK,
    # This task's actor tracks reference clips: actor + motions + selector.
    executor="deploy.motion_20261119.reference.hope_g1.motion_executor.MotionExecutor",
    stage="hope_multi_motion",
    iteration=policy.iteration,
    anchor=motion["anchor_body_name"],
    motions=[p.name for p in paths],
    hope=dict(
      collision_source=str(urdf),
      self_collisions=self_collisions,
      motion=motion,
      racket=racket,
      hit_types=hit_types,
      strike_frames=strike_frames,
      return_seconds=motion["return_to_start_steps"] * control_dt,
      max_start_frame=motion["random_start_frame_max"],
      # Isaac never starts a swing later than strike_frame - lead.
      min_lead_frames=motion.get("random_start_frame_min_lead_frames"),
      dual_hand=dual_hand,
      racket_hands=racket_hands,
    ),
    actor_terms=actor_terms,
  )
  return HopeDeployment(
    settings=settings,
    model=model,
    motions=paths,
    tracker=None,
    actor=policy,
    checkpoint=checkpoint,
    tracker_path=None,
  )

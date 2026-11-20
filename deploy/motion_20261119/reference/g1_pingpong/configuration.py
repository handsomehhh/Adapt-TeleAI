"""Direct policy/configuration loading for G1 simulation and hardware."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from deploy.motion_20261119.reference.g1_pingpong.core import ACTOR_TERMS, TRACKER_TERMS
from deploy.motion_20261119.reference.g1_pingpong.policy import load_policy, run_directory, training_config


@dataclass
class Deployment:
  settings: dict[str, Any]
  model: Any
  motions: list[Path]
  tracker: Any
  actor: Any = None
  checkpoint: Path | None = None
  tracker_path: Path | None = None

  def report(self):
    return {
      "checkpoint": str(self.checkpoint),
      "tracker": str(self.tracker_path),
      "task": self.settings["task"],
      "stage": self.settings["stage"],
      "num_motions": len(self.motions),
      "tracker_input": 132,
      "actor_input": 186 if self.actor else 132,
    }


TASKS = {
  "g1_pingpong_receive_stage2_tele_gmr_hard": "Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive",
  "g1_pingpong_receive_stage2_tele_gmr": "Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive",
  "g1_pingpong_receive_stage2": "Mjlab-PingPong-Receive-G1-Stage2-Adaptive",
  "g1_pingpong_tracking_stage1_random_dt": "Mjlab-PingPong-Tracking-G1-Stage1-RandomDt",
  "g1_pingpong_tracking_stage1": "Mjlab-PingPong-Tracking-G1-Stage1-RandomDt",
}


def add_policy_arguments(parser):
  parser.add_argument(
    "--config", type=Path, help="Deployment YAML; CLI paths override it"
  )
  parser.add_argument(
    "--checkpoint",
    "--policy",
    dest="checkpoint",
    type=Path,
    required=False,
    help="Training model_N.pt or an existing JIT",
  )
  parser.add_argument(
    "--task", help="G1 task ID; inferred from run metadata if omitted"
  )
  parser.add_argument("--tracker", type=Path, help="Stage 2 frozen tracker path")
  parser.add_argument("--env-config", type=Path, help="Saved params/env.yaml override")
  parser.add_argument(
    "--agent-config", type=Path, help="Saved params/agent.yaml override"
  )
  parser.add_argument("--xml-path", type=Path, help="Existing shared MuJoCo XML")
  motions = parser.add_mutually_exclusive_group()
  motions.add_argument("--motion-dir", type=Path, help="Relocated motion library")
  motions.add_argument("--motion-file", type=Path, help="Use one reference motion")


def from_arguments(args):
  options = (yaml.safe_load(args.config.read_text()) or {}) if args.config else {}
  for name in (
    "checkpoint",
    "task",
    "tracker",
    "env_config",
    "agent_config",
    "motion_dir",
    "motion_file",
    "xml_path",
  ):
    value = getattr(args, name)
    if value is not None:
      options[name] = value
  if args.motion_file is not None:
    options.pop("motion_dir", None)
  elif args.motion_dir is not None:
    options.pop("motion_file", None)
  if not options.get("checkpoint"):
    raise ValueError("Specify --checkpoint/--policy or checkpoint in --config")
  return load_deployment(**options)


def infer_task(checkpoint, agent_config=None):
  agent = (
    Path(agent_config)
    if agent_config
    else run_directory(checkpoint) / "params/agent.yaml"
  )
  experiment = training_config(agent).get("experiment_name", "")
  if experiment in TASKS:
    return TASKS[experiment]
  raise ValueError(f"Cannot infer task from {experiment!r}; specify --task")


def load_deployment(
  checkpoint,
  *,
  task=None,
  tracker=None,
  env_config=None,
  agent_config=None,
  motion_dir=None,
  motion_file=None,
  xml_path=None,
):
  task = task or infer_task(checkpoint, agent_config)
  # Read the selected run configuration; build the shared robot in memory.
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.scene import Scene
  from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommandCfg
  from mjlab.tasks.registry import load_env_cfg

  checkpoint = Path(checkpoint).resolve()
  if "PingPong" not in task or "G1" not in task:
    raise ValueError("Only the 29-DoF G1 ping-pong tasks are supported")
  run = run_directory(checkpoint)
  env_config = Path(env_config) if env_config else run / "params/env.yaml"
  saved = training_config(env_config)
  cfg = load_env_cfg(task, play=True)
  stage2 = "Stage2" in task
  groups = ["actor", "tracker"] if stage2 else ["actor"]
  for group in groups:
    saved_group = saved["observations"][group]
    if saved_group.get("history_length") or not saved_group["concatenate_terms"]:
      raise ValueError(f"Unsupported observation group configuration: {group}")
    terms = saved_group["terms"]
    expected = ACTOR_TERMS if stage2 and group == "actor" else TRACKER_TERMS
    if list(terms) != list(expected):
      raise ValueError(f"Unsupported observation order in {group}")
    for name, term in terms.items():
      if any(term.get(k) is not None for k in ("clip", "scale")) or any(
        term.get(k) for k in ("history_length", "delay_max_lag")
      ):
        raise ValueError(f"Unsupported observation transformation: {group}.{name}")
      expected_func = cfg.observations[group].terms[name].func
      qualified = expected_func.__module__ + "." + expected_func.__name__
      if term["func"] != qualified:
        raise ValueError(f"Observation function differs: {group}.{name}")
  action = saved["actions"]["joint_pos"]
  if action.get("clip") is not None or not action["use_default_offset"]:
    raise ValueError("Unsupported action clipping/offset configuration")
  configured_action = cfg.actions["joint_pos"]
  assert isinstance(configured_action, JointPositionActionCfg)
  names = list(configured_action.actuator_names)
  if list(action["actuator_names"]) != names:
    raise ValueError("Policy joint order differs from the G1 asset")
  robot_cfg = cfg.scene.entities["robot"]
  init = saved["scene"]["entities"]["robot"]["init_state"]
  for key, value in init.items():
    setattr(robot_cfg.init_state, key, value)
  actuators = saved["scene"]["entities"]["robot"]["articulation"]["actuators"]
  assert robot_cfg.articulation is not None
  for actuator, record in zip(robot_cfg.articulation.actuators, actuators, strict=True):
    if list(actuator.target_names_expr) != record["target_names_expr"]:
      raise ValueError("Actuator mapping differs")
    if record.get("delay_max_lag", 0):
      raise ValueError("Actuator delay is not supported by the independent runtime")
    for key in (
      "stiffness",
      "damping",
      "effort_limit",
      "armature",
      "frictionloss",
      "viscous_damping",
    ):
      setattr(actuator, key, record.get(key))
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  joint_ids = [model.joint("robot/" + name).id for name in names]
  qpos_ids = model.jnt_qposadr[joint_ids]
  act_ids = []
  for j in joint_ids:
    ids = np.flatnonzero(model.actuator_trnid[:, 0] == j)
    if len(ids) != 1:
      raise ValueError("Expected exactly one position actuator per joint")
    act_ids.append(int(ids[0]))
  input_dim, output_dim = (186, 30) if stage2 else (132, 29)
  actor = load_policy(checkpoint, input_dim, output_dim, agent_config)
  tracker_path = Path(tracker).resolve() if tracker else run / "tracker.pt"
  tracker_policy = load_policy(tracker_path, 132, 29) if stage2 else actor
  iteration = actor.iteration
  motion = saved["commands"]["motion"]
  paths = [Path(p) for p in motion["motion_files"]]
  if motion_file is not None:
    paths = [Path(motion_file).resolve()]
  elif motion_dir is not None:
    paths = [Path(motion_dir).resolve() / p.name for p in paths]
  if not paths or motion.get("align_heading_to_frame") != "none":
    raise ValueError("Expected prepared, named G1 motions with no heading realignment")
  if len({p.name for p in paths}) != len(paths):
    raise ValueError("Duplicate motion filenames")
  # Validate selected motion files without copying them.
  from deploy.motion_20261119.reference.g1_pingpong.core import Motion

  for path in paths:
    Motion(path, names, "torso_link")
  saved_ball = saved["commands"].get("ball", {})
  # Older runs predate lateral feeds/difficulty ramp fields. Their absent
  # lateral range means zero lateral velocity, not today's Hard task default.
  ball = (
    asdict(ReceivingBallCommandCfg(resampling_time_range=(1e9, 1e9))) if stage2 else {}
  )
  compatibility_defaults = [key for key in ball if key not in saved_ball]
  ball.update(saved_ball)
  if stage2 and list(ball["prediction_times"]) != [0.1, 0.2, 0.3]:
    raise ValueError("Unsupported ball prediction shape")
  prefixes = tuple(action.get("hit_arm_joint_prefixes", []))
  scales = [
    action.get("hit_arm_scale", 0)
    if n.startswith(prefixes)
    else action.get("residual_scale", 0)
    for n in names
  ]
  dt = float(saved["sim"]["mujoco"]["timestep"])
  control_dt = dt * saved["decimation"]
  if not np.isclose(control_dt, motion["fixed_dt"]):
    raise ValueError("Reference and control fixed dt must agree")
  scale = action["scale"]
  default_q = data.qpos[qpos_ids].tolist()
  settings = {
    "task": task,
    "stage": 2 if stage2 else 1,
    "iteration": iteration,
    "control_dt": control_dt,
    "physics_dt": dt,
    "decimation": saved["decimation"],
    "sim_options": saved["sim"]["mujoco"],
    "episode_length_s": saved["episode_length_s"],
    "termination_thresholds": {
      name: term.get("params", {}).get("threshold")
      for name, term in saved["terminations"].items()
      if term.get("params", {}).get("threshold") is not None
    },
    "warmdown_s": saved["terminations"]
    .get("motion_finished", {})
    .get("params", {})
    .get("warmdown_s", 0.25),
    "motions": [p.name for p in paths],
    "anchor": "torso_link",
    "tracker_terms": TRACKER_TERMS,
    "actor_terms": ACTOR_TERMS if stage2 else TRACKER_TERMS,
    "robot": {
      "joint_names": names,
      "default_q": default_q,
      "kp": model.actuator_gainprm[act_ids, 0].tolist(),
      "kd": (-model.actuator_biasprm[act_ids, 2]).tolist(),
      "torque_limits": model.actuator_forcerange[act_ids, 1].tolist(),
      "joint_limits": model.jnt_range[joint_ids].tolist(),
      "action_scale": [scale[n] for n in names]
      if isinstance(scale, dict)
      else [scale] * 29,
    },
    "action": {
      "residual_scales": scales,
      "residual_clip": action.get("residual_clip", 2),
      "min_speed": action.get("min_speed", 0.5),
      "max_speed": action.get("max_speed", 2.0),
    },
    "ball": ball,
    "compatibility_defaults": compatibility_defaults,
  }
  return Deployment(
    settings=settings,
    model=Path(xml_path).resolve() if xml_path else model,
    motions=[p.resolve() for p in paths],
    tracker=tracker_policy,
    actor=actor if stage2 else None,
    checkpoint=checkpoint,
    tracker_path=tracker_path if stage2 else checkpoint,
  )

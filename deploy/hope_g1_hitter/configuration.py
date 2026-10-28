"""Load the mjlab G1 hitter (104 -> 29) checkpoint and its exported contract."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from deploy.g1_pingpong.configuration import Deployment
from deploy.g1_pingpong.pd_gains import deployment_gains
from deploy.g1_pingpong.policy import TrainingYamlLoader
from deploy.hope_g1_hitter.policy import HitterPolicy

TASK = "HOPEHitterPingPongG1"
EXPERIMENT = "unitree_g1_hitter_pingpong_mjlab"
# (actor term name, evaluated function, width), in g1_hitter_pure_v1 order.
OBSERVATIONS = (
  ("base_ang_vel", "base_ang_vel", 3),
  ("joint_pos", "joint_pos_rel", 29),
  ("joint_vel", "joint_vel_rel", 29),
  ("actions", "applied_last_action", 29),
  ("projected_gravity", "projected_gravity_mocap", 3),
  ("base_forward_xy", "base_forward_xy_mocap", 2),
  ("base_target_delta_xy", "base_target_delta_xy_mocap", 2),
  ("racket_target_rel_base", "racket_target_rel_base_mocap", 3),
  ("racket_target_vel_w", "racket_target_vel_w", 3),
  ("time_to_strike", "time_to_strike", 1),
)


class MjlabYamlLoader(TrainingYamlLoader):
  """Also read mjlab's ``!!python/name`` references as plain strings."""


MjlabYamlLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
)


def read_yaml(path):
  return yaml.load(Path(path).read_text(), Loader=MjlabYamlLoader)


@dataclass
class HitterDeployment(Deployment):
  def report(self):
    return dict(
      task=TASK,
      controller="deploy.hope_g1_hitter.controller.HitterController",
      checkpoint=str(self.checkpoint),
      iteration=self.settings["iteration"],
      actor_input=sum(width for *_, width in OBSERVATIONS),
      actor_output=29,
      control_dt=self.settings["control_dt"],
      motions=[p.name for p in self.motions],
      normalization="none",
      world_frame=self.settings["world_frame"],
      simulation_world_translation=self.settings["simulation_world_translation"],
    )


def validate_training(env, agent):
  """Reject any checkpoint whose saved actor/action contract is not 104D."""
  if agent.get("experiment_name") != EXPERIMENT:
    raise ValueError(f"Expected the {EXPERIMENT} training task")
  actor = agent["actor"]
  if (
    actor["class_name"] != "MLPModel"
    or actor["activation"] != "elu"
    or actor["obs_normalization"]
    or actor.get("rnn_type") is not None
    or agent.get("clip_actions") is not None
  ):
    raise ValueError("Expected an unnormalized feed-forward ELU MLP actor")
  group = env["observations"]["actor"]
  if not group["concatenate_terms"] or group.get("history_length"):
    raise ValueError("Unsupported hitter observation history/layout")
  terms = group["terms"]
  if [(n, t["params"]["name"]) for n, t in terms.items()] != [
    (n, f) for n, f, _ in OBSERVATIONS
  ]:
    raise ValueError("Hitter actor observation order differs from 104-D contract")
  for name, term in terms.items():
    if (
      any(term.get(k) is not None for k in ("clip", "scale"))
      or term.get("history_length")
      or term.get("delay_min_lag")
      or term.get("delay_max_lag")
    ):
      raise ValueError(f"Unsupported hitter observation transformation: {name}")
  if env["actions"]["joint_pos"].get("clip") is not None:
    raise ValueError("Unsupported hitter action clip")
  control_dt = float(env["sim"]["mujoco"]["timestep"]) * env["decimation"]
  if not np.isclose(control_dt, 0.02):
    raise ValueError("Hitter task must run at 50Hz")
  return control_dt


def load_deployment(artifact_dir, *, ball_config, world_config):
  from deploy.hope_g1.scene import table_cfg
  from deploy.hope_g1.world import (
    mujoco_ball_config,
    simulation_translation,
    validate_table_world,
  )
  from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
    get_g1_pingpong_robot_cfg,
  )
  from mjlab.scene import Scene, SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  run = Path(artifact_dir).resolve()
  checkpoints = sorted(run.glob("model_*.pt"))
  if len(checkpoints) != 1:
    raise ValueError(f"Expected exactly one model_*.pt in {run}")
  checkpoint = checkpoints[0]
  env, agent = read_yaml(run / "params/env.yaml"), read_yaml(run / "params/agent.yaml")
  control_dt = validate_training(env, agent)
  contract = json.loads((run / "contract.json").read_text())
  if contract["actor_obs_contract"] != "g1_hitter_pure_v1" or not np.isclose(
    contract["control_dt"], control_dt
  ):
    raise ValueError("contract.json does not describe this hitter checkpoint")
  names = list(contract["joint_names"])
  default = np.asarray(contract["default_q"], dtype=float)
  saved_default = env["scene"]["entities"]["robot"]["init_state"]["joint_pos"]
  if len(names) != 29 or not np.allclose(
    default, [saved_default[n] for n in names], atol=1e-6
  ):
    raise ValueError("contract.json default pose differs from the saved env.yaml")
  kp, kd = np.asarray(contract["kp"]), np.asarray(contract["kd"])
  # The shared runtime executes the literal PD table; require agreement up front.
  execution = deployment_gains(names)
  if not np.allclose(kp, execution[0]) or not np.allclose(kd, execution[1]):
    raise ValueError("Hitter training PD differs from the shared execution PD table")
  paths = [run / "motions" / clip["file"] for clip in contract["clips"]]
  validate_table_world(ball_config)
  translation = simulation_translation(world_config)
  simulation_ball = mujoco_ball_config(ball_config, translation)
  robot = get_g1_pingpong_robot_cfg()
  robot.init_state.joint_pos = dict(zip(names, default.tolist(), strict=True))
  robot.init_state.pos = (0.0, 0.0, float(contract["ready_height"]))
  model = Scene(
    SceneCfg(
      entities={"robot": robot, "table": table_cfg(simulation_ball)},
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    device="cpu",
  ).compile()
  joints = [model.joint("robot/" + n).id for n in names]
  torque = np.asarray(contract["effort"], dtype=float)
  for j, limit in zip(joints, torque, strict=True):
    a = np.flatnonzero(model.actuator_trnid[:, 0] == j)
    if len(a) != 1 or not np.isclose(model.actuator_forcerange[a[0], 1], limit):
      raise ValueError("Simulation actuator effort differs from the training contract")
  hard = np.stack([contract["hard_lower"], contract["hard_upper"]], axis=1)
  if not np.allclose(model.jnt_range[joints], hard, atol=1e-6):
    raise ValueError("Simulation joint range differs from the training contract")
  policy = HitterPolicy(checkpoint, agent)
  dt = float(env["sim"]["mujoco"]["timestep"])
  settings = dict(
    ball=dict(ball_config),
    world_frame="hope_table",
    simulation_world_translation=translation.tolist(),
    simulation_ball=simulation_ball,
    task=TASK,
    stage="hope_hitter",
    iteration=policy.iteration,
    control_dt=control_dt,
    physics_dt=dt,
    decimation=env["decimation"],
    sim_options=dict(integrator="implicitfast", timestep=dt, iterations=20),
    anchor="torso_link",
    motions=[p.name for p in paths],
    robot=dict(
      joint_names=names,
      default_q=default.tolist(),
      default_dq=[0.0] * 29,
      kp=kp.tolist(),
      kd=kd.tolist(),
      torque_limits=torque.tolist(),
      joint_limits=hard.tolist(),
      action_scale=contract["action_scale"],
    ),
    hitter=contract,
  )
  return HitterDeployment(
    settings=settings,
    model=model,
    motions=paths,
    tracker=None,
    actor=policy,
    checkpoint=checkpoint,
    tracker_path=None,
  )

"""Task configurations kept on the standard mjlab train/play/logging path."""

from copy import deepcopy
from pathlib import Path

from mjlab.asset_zoo.robots.agibot_a3.a3_constants import (
  A3_ACTION_SCALE,
  A3_ANCHOR_BODY,
  A3_FEET_BODIES,
  A3_JOINT_NAMES,
  A3_TRACKED_BODIES,
  get_a3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.a3_pingpong.actions import (
  FrozenTrackerActionCfg,
  OrderedJointPositionActionCfg,
)
from mjlab.tasks.a3_pingpong.ball import (
  ReceivingBallCommandCfg,
  ball_approach_reward,
  ball_hit_reward,
  ball_intercept_reward,
  ball_landing_reward,
  ball_net_reward,
  ball_normal_reward,
  ball_observation,
)
from mjlab.tasks.a3_pingpong.runner import A3OnPolicyRunner
from mjlab.tasks.a3_pingpong.scene import table_cfg
from mjlab.tasks.a3_pingpong.tracking import (
  A3MotionCommandCfg,
  ball_finished,
  hit_arm_tracking,
  joint_action_history,
  motion_speed,
  residual_l2,
  speed_regularization,
  wrist_tracking,
)
from mjlab.tasks.adapt_tennis import mdp
from mjlab.tasks.adapt_tennis.config.g1.rl_cfg import (
  unitree_g1_adapt_tennis_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

DATA_DIR = Path(__file__).resolve().parents[4] / "dataset" / "a3_pingpong" / "motions"


def tracker_observations():
  joints = SceneEntityCfg("robot", joint_names=A3_JOINT_NAMES, preserve_order=True)
  return {
    "command": ObservationTermCfg(
      func=common.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "projected_gravity": ObservationTermCfg(
      func=common.projected_gravity, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=common.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1)
    ),
    "joint_pos": ObservationTermCfg(
      func=common.joint_pos_rel,
      params={"asset_cfg": joints, "biased": True},
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=common.joint_vel_rel,
      params={"asset_cfg": joints},
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "actions": ObservationTermCfg(func=joint_action_history),
    "motion_dt_sample": ObservationTermCfg(func=mdp.motion_dt_sample),
  }


def a3_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor = tracker_observations()
  critic = deepcopy(actor)
  critic.update(
    {
      "base_lin_vel": ObservationTermCfg(func=common.base_lin_vel),
      "body_pos": ObservationTermCfg(
        func=mdp.robot_body_pos_b, params={"command_name": "motion"}
      ),
      "body_ori": ObservationTermCfg(
        func=mdp.robot_body_ori_b, params={"command_name": "motion"}
      ),
    }
  )
  rewards = {
    "motion_anchor_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_anchor_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=1,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 3.14},
    ),
    "motion_wrist": RewardTermCfg(func=wrist_tracking, weight=0.5),
    "motion_hit_arm": RewardTermCfg(func=hit_arm_tracking, weight=0.5),
    "action_rate": RewardTermCfg(func=common.action_rate_l2, weight=-0.02),
    "joint_limit": RewardTermCfg(func=common.joint_pos_limits, weight=-2.0),
  }
  events = {
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.005, 0.005)},
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1 if play else 1024,
      env_spacing=5.0,
      entities={"robot": get_a3_robot_cfg()},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor,
        concatenate_terms=True,
        enable_corruption=not play,
      ),
      "critic": ObservationGroupCfg(
        terms=critic,
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions={
      "joint_pos": OrderedJointPositionActionCfg(
        entity_name="robot",
        actuator_names=A3_JOINT_NAMES,
        preserve_order=True,
        scale=A3_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={
      "motion": A3MotionCommandCfg(
        entity_name="robot",
        motion_directory=str(DATA_DIR),
        joint_names=A3_JOINT_NAMES,
        body_names=A3_TRACKED_BODIES,
        anchor_body_name=A3_ANCHOR_BODY,
        align_heading_to_frame="none",
        resampling_time_range=(1e9, 1e9),
        debug_vis=play,
        sampling_mode="start" if play else "adaptive",
        dynamic_dt_enabled=True,
        random_dt_training_enabled=not play,
        fixed_dt=0.02,
        dt_delta_range=(-0.01, 0.02),
        joint_position_range=(0, 0) if play else (-0.03, 0.03),
        pose_range={} if play else {"roll": (-0.03, 0.03), "pitch": (-0.03, 0.03)},
        velocity_range={},
      )
    },
    events={} if play else events,
    rewards=rewards,
    terminations={
      "motion_finished": TerminationTermCfg(
        func=mdp.motion_reached_last_frame,
        time_out=True,
        params={"command_name": "motion", "warmdown_s": 0.25},
      ),
      "anchor_height": TerminationTermCfg(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
      ),
      "anchor_orientation": TerminationTermCfg(
        func=mdp.bad_anchor_ori,
        params={
          "command_name": "motion",
          "threshold": 0.8,
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "feet_height": TerminationTermCfg(
        func=mdp.bad_motion_body_pos_z_only,
        params={
          "command_name": "motion",
          "threshold": 0.25,
          "body_names": A3_FEET_BODIES,
        },
      ),
    },
    sim=SimulationCfg(
      nconmax=64,
      njmax=256,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=10),
    ),
    decimation=4,
    episode_length_s=4.0,
    scale_rewards_by_motion_dt=False,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=A3_ANCHOR_BODY,
      distance=3.5,
      elevation=-15,
      azimuth=135,
    ),
  )


def a3_runner_cfg(stage: int = 1):
  cfg = unitree_g1_adapt_tennis_ppo_runner_cfg()
  cfg.experiment_name = (
    "a3_pingpong_tracking_stage1_random_dt"
    if stage == 1
    else "a3_pingpong_receive_stage2"
  )
  cfg.save_interval = 500
  cfg.max_iterations = 10000
  cfg.upload_model = False
  cfg.logger = "tensorboard"
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.5
  cfg.algorithm.entropy_coef = 0.002
  return cfg


def a3_receiving_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = a3_tracking_env_cfg(play)
  cfg.scene.entities["table"] = table_cfg()
  cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  cfg.viewer.lookat = (1.2, 0.0, 0.9)
  cfg.viewer.distance = 4.6
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 125.0
  motion = cfg.commands["motion"]
  assert isinstance(motion, A3MotionCommandCfg)
  motion.sampling_mode = "start"
  motion.random_dt_training_enabled = False
  cfg.observations["tracker"] = deepcopy(cfg.observations["actor"])
  cfg.observations["tracker"].enable_corruption = False
  for name in ("actor", "critic"):
    cfg.observations[name].terms["ball"] = ObservationTermCfg(func=ball_observation)
    cfg.observations[name].terms["phase"] = ObservationTermCfg(func=mdp.motion_phase)
    cfg.observations[name].terms["high_level_action"] = ObservationTermCfg(
      func=common.last_action
    )
  cfg.actions["joint_pos"] = FrozenTrackerActionCfg(
    entity_name="robot",
    actuator_names=A3_JOINT_NAMES,
    preserve_order=True,
    scale=A3_ACTION_SCALE,
    use_default_offset=True,
    hit_arm_scale=1.0,
  )
  cfg.commands["ball"] = ReceivingBallCommandCfg(
    resampling_time_range=(1e9, 1e9),
    debug_vis=True,
    position_noise=0.0 if play else 0.005,
    velocity_noise=0.0 if play else 0.03,
  )
  cfg.terminations.pop("motion_finished")
  cfg.terminations["time_out"] = TerminationTermCfg(func=common.time_out, time_out=True)
  cfg.terminations["ball_finished"] = TerminationTermCfg(func=ball_finished)
  cfg.episode_length_s = 3.0
  for name, term in cfg.rewards.items():
    if name.startswith("motion_"):
      term.weight *= 0.3
  cfg.rewards.update(
    {
      "ball_approach": RewardTermCfg(func=ball_approach_reward, weight=5.0),
      "ball_intercept": RewardTermCfg(func=ball_intercept_reward, weight=3.0),
      "ball_normal": RewardTermCfg(func=ball_normal_reward, weight=1.0),
      "ball_hit": RewardTermCfg(func=ball_hit_reward, weight=100.0),
      "ball_net": RewardTermCfg(func=ball_net_reward, weight=100.0),
      "ball_landing": RewardTermCfg(func=ball_landing_reward, weight=200.0),
      "residual": RewardTermCfg(func=residual_l2, weight=-0.05),
      "speed": RewardTermCfg(func=speed_regularization, weight=-0.02),
    }
  )
  # Ball task rewards integrate real time: physical ball time never changes
  # when the policy speeds up/slows down the reference motion.
  cfg.scale_rewards_by_motion_dt = False
  cfg.metrics = {
    "motion_speed": MetricsTermCfg(func=motion_speed),
    "residual_l2": MetricsTermCfg(func=residual_l2),
  }
  return cfg


def register_tasks():
  register_mjlab_task(
    task_id="Mjlab-PingPong-Tracking-A3-Stage1-RandomDt",
    env_cfg=a3_tracking_env_cfg(),
    play_env_cfg=a3_tracking_env_cfg(play=True),
    rl_cfg=a3_runner_cfg(),
    runner_cls=A3OnPolicyRunner,
  )
  register_mjlab_task(
    task_id="Mjlab-PingPong-Receive-A3-Stage2-Adaptive",
    env_cfg=a3_receiving_env_cfg(),
    play_env_cfg=a3_receiving_env_cfg(play=True),
    rl_cfg=a3_runner_cfg(stage=2),
    runner_cls=A3OnPolicyRunner,
  )

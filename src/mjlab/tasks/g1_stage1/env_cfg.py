"""Standalone 29-DoF G1 random-speed tracker; no ball or old checkpoint."""

import os
from copy import deepcopy
from pathlib import Path

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_ANCHOR_BODY,
  G1_PINGPONG_FEET_BODIES,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_TRACKED_BODIES,
  get_g1_pingpong_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .actions import OrderedJointPositionActionCfg
from .mdp.legacy_schema import LEGACY_G1_BODY_NAMES, LEGACY_G1_JOINT_NAMES
from .mdp.tracking import (
  Stage1MotionCommandCfg,
  hit_arm_tracking,
  racket_tracking,
  wrist_tracking,
)
from .training_profile import training_profile

DATA_DIR = Path(__file__).resolve().parents[4] / "dataset" / "g1_stage1" / "motions"


def tracker_observations():
  joints = SceneEntityCfg(
    "robot", joint_names=G1_PINGPONG_JOINT_NAMES, preserve_order=True
  )
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
    "actions": ObservationTermCfg(func=common.last_action),
    "motion_dt_sample": ObservationTermCfg(func=mdp.motion_dt_sample),
  }


def g1_stage1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
    "waist_action_rate": RewardTermCfg(
      func=common.action_rate_l2,
      weight=-0.04,
      params={
        "action_name": "joint_pos",
        "exclude_joint_names": tuple(
          name for name in G1_PINGPONG_JOINT_NAMES if not name.startswith("waist_")
        ),
      },
    ),
    "joint_limit": RewardTermCfg(func=common.joint_pos_limits, weight=-2.0),
  }
  profile = training_profile().get("tracker", {})
  for name, weight in profile.get("reward_weights", {}).items():
    if name not in rewards:
      raise ValueError(f"Unknown tracker reward: {name}")
    rewards[name].weight = float(weight)
  if profile.get("racket_center_weight", 0) > 0:
    rewards["racket_center"] = RewardTermCfg(
      func=racket_tracking,
      weight=profile["racket_center_weight"],
      params={"std": profile.get("racket_center_std", 0.08)},
    )
  if profile.get("racket_normal_weight", 0) > 0:
    rewards["racket_normal"] = RewardTermCfg(
      func=racket_tracking,
      weight=profile["racket_normal_weight"],
      params={"std": profile.get("racket_normal_std", 0.25), "normal": True},
    )
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
      entities={"robot": get_g1_pingpong_robot_cfg()},
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
        actuator_names=G1_PINGPONG_JOINT_NAMES,
        preserve_order=True,
        scale=G1_PINGPONG_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={
      "motion": Stage1MotionCommandCfg(
        entity_name="robot",
        motion_directory=os.environ.get("ADAPT_G1_MOTION_DIR", str(DATA_DIR)),
        joint_names=G1_PINGPONG_JOINT_NAMES,
        legacy_body_names=LEGACY_G1_BODY_NAMES,
        legacy_joint_names=LEGACY_G1_JOINT_NAMES,
        body_names=G1_PINGPONG_TRACKED_BODIES,
        anchor_body_name=G1_PINGPONG_ANCHOR_BODY,
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
          "body_names": G1_PINGPONG_FEET_BODIES,
        },
      ),
    },
    sim=SimulationCfg(
      nconmax=256,
      njmax=1024,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=10),
    ),
    decimation=4,
    episode_length_s=1e9 if play else 4.0,
    scale_rewards_by_motion_dt=False,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=G1_PINGPONG_ANCHOR_BODY,
      distance=3.5,
      elevation=-15,
      azimuth=135,
    ),
  )

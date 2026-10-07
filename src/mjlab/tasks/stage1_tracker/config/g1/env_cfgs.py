"""Generic G1 stage-1 motion tracker configuration.

The task follows the project's adaptive-dt stage-1 recipe while keeping the
reward contract robot and motion-source agnostic. Motion clips are read by
name from a recursive directory, so a dataset can be split into arbitrary
subdirectories without generating a manifest first.
"""

from __future__ import annotations

import os

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_ANCHOR_BODY,
  G1_PINGPONG_FEET_BODIES,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_TRACKED_BODIES,
  get_g1_pingpong_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.a3_pingpong.actions import OrderedJointPositionActionCfg
from mjlab.tasks.a3_pingpong.config import a3_tracking_env_cfg
from mjlab.tasks.a3_pingpong.tracking import A3MotionCommandCfg
from mjlab.tasks.adapt_tennis import mdp

DEFAULT_MOTION_DIRECTORY = "/data_zcy/zcy/datasets/motion_data_used_g1"
DEFAULT_MAX_MOTION_CLIPS = 1024
DEFAULT_MAX_MOTION_FRAMES = 4096


def _motion_directory() -> str:
  return os.environ.get("ADAPT_STAGE1_MOTION_DIR", DEFAULT_MOTION_DIRECTORY)


def unitree_g1_stage1_tracker_env_cfg(
  *, play: bool = False, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  """Build the generic stage-1 tracker for the validated 29-DoF G1 asset.

  ``ADAPT_STAGE1_MOTION_DIR`` and ``ADAPT_STAGE1_NUM_ENVS`` are convenient
  deployment overrides; both can also be replaced through the regular Tyro
  nested config flags used by ``train``.
  """
  cfg = a3_tracking_env_cfg(play=play)

  # Replace the A3 robot and all A3-specific ordering contracts.
  cfg.scene.entities["robot"] = get_g1_pingpong_robot_cfg()
  ordered_joints = SceneEntityCfg(
    "robot", joint_names=G1_PINGPONG_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for term_name in ("joint_pos", "joint_vel"):
      term = group.terms.get(term_name)
      if term is not None:
        term.params["asset_cfg"] = ordered_joints

  action = cfg.actions["joint_pos"]
  assert isinstance(action, OrderedJointPositionActionCfg)
  action.actuator_names = G1_PINGPONG_JOINT_NAMES
  action.scale = G1_PINGPONG_ACTION_SCALE

  motion = cfg.commands["motion"]
  assert isinstance(motion, A3MotionCommandCfg)
  motion.motion_file = ""
  motion.motion_files = ()
  motion.motion_directory = _motion_directory()
  motion.max_motion_clips = int(
    os.environ.get("ADAPT_STAGE1_MAX_MOTIONS", str(DEFAULT_MAX_MOTION_CLIPS))
  )
  motion.max_motion_frames = int(
    os.environ.get("ADAPT_STAGE1_MAX_FRAMES", str(DEFAULT_MAX_MOTION_FRAMES))
  )
  motion.joint_names = G1_PINGPONG_JOINT_NAMES
  motion.body_names = G1_PINGPONG_TRACKED_BODIES
  motion.anchor_body_name = G1_PINGPONG_ANCHOR_BODY
  motion.align_heading_to_frame = "none"
  motion.debug_vis = play
  # Starting every rollout at frame zero avoids treating a random near-end
  # frame as a one-step episode.  This is especially important for PPO's
  # episode statistics and gives the tracker a full clip horizon to learn
  # from.  Adaptive in-clip sampling remains available for experiments.
  motion.sampling_mode = "start" if play else os.environ.get(
    "ADAPT_STAGE1_SAMPLING_MODE", "start"
  )
  motion.random_dt_training_enabled = not play
  motion.joint_position_range = (0.0, 0.0) if play else (-0.03, 0.03)
  motion.pose_range = {} if play else {"roll": (-0.03, 0.03), "pitch": (-0.03, 0.03)}
  motion.velocity_range = {}

  # Strike-specific bonuses from the ping-pong task are deliberately absent.
  # The remaining root/body pose and velocity terms train a reusable tracker
  # over any G1 motion collection.
  cfg.rewards.pop("motion_wrist", None)
  cfg.rewards.pop("motion_hit_arm", None)
  cfg.terminations["feet_height"].params["body_names"] = G1_PINGPONG_FEET_BODIES
  cfg.viewer.body_name = G1_PINGPONG_ANCHOR_BODY
  # The 29-DoF G1 has more self-contact candidates than the A3 template.
  # Leave enough MuJoCo workspace for randomized tracking rollouts.
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 1024

  if num_envs is None:
    configured_num_envs = int(os.environ.get("ADAPT_STAGE1_NUM_ENVS", "1000"))
    num_envs = 1 if play else configured_num_envs
  if num_envs < 1:
    raise ValueError(f"num_envs must be positive, got {num_envs}")
  cfg.scene.num_envs = num_envs

  # Use the dynamic-time termination, which is aware of interpolated clip time.
  cfg.terminations["motion_finished"] = TerminationTermCfg(
    func=mdp.motion_reached_last_frame,
    time_out=True,
    params={"command_name": "motion", "warmdown_s": 0.25},
  )

  # Keep play mode deterministic and avoid mutating the training configuration
  # when this helper is called repeatedly by the task registry.
  if play:
    cfg.events = {}
    for group_name in ("actor", "critic"):
      group = cfg.observations.get(group_name)
      if isinstance(group, ObservationGroupCfg):
        group.enable_corruption = False

  return cfg

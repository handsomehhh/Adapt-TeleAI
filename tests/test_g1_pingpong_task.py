"""Configuration contract for the 29-DoF G1 two-stage ping-pong task."""

from pathlib import Path

import pytest

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_TRACKED_BODIES,
)
from mjlab.tasks.a3_pingpong.actions import FrozenTrackerActionCfg
from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommandCfg
from mjlab.tasks.g1_pingpong.config import (
  MOTION_FILES,
  STAGE1_TASK,
  STAGE2_TASK,
  TELE_GMR_DATA_ROOT,
  TELE_GMR_HARD_STAGE2_TASK,
  TELE_GMR_MOTION_FILES,
  TELE_GMR_STAGE1_TASK,
  TELE_GMR_STAGE2_TASK,
  TELE_GMR_TRACKER,
  validate_motion_dataset,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

pytestmark = pytest.mark.skipif(
  not MOTION_FILES or not TELE_GMR_MOTION_FILES,
  reason="Optional G1 reference datasets are not installed.",
)


def test_g1_pingpong_motion_contract_is_exact_and_frozen():
  assert validate_motion_dataset() == MOTION_FILES
  assert len(MOTION_FILES) == 43
  assert len({Path(path).name for path in MOTION_FILES}) == 43


def test_g1_stage1_uses_29dof_order_and_canonical_motion_set():
  cfg = load_env_cfg(STAGE1_TASK)
  action = cfg.actions["joint_pos"]
  motion = cfg.commands["motion"]

  assert action.actuator_names == G1_PINGPONG_JOINT_NAMES
  assert action.scale == G1_PINGPONG_ACTION_SCALE
  assert motion.motion_files == MOTION_FILES
  assert motion.motion_directory == ""
  assert motion.joint_names == G1_PINGPONG_JOINT_NAMES
  assert motion.body_names == G1_PINGPONG_TRACKED_BODIES
  assert motion.align_heading_to_frame == "none"
  assert motion.random_dt_training_enabled
  assert cfg.scene.num_envs == 1024
  assert load_rl_cfg(STAGE1_TASK).experiment_name == (
    "g1_pingpong_tracking_stage1_random_dt"
  )


def test_g1_tele_gmr_stage1_only_replaces_the_verified_motion_set():
  assert (
    validate_motion_dataset(
      TELE_GMR_DATA_ROOT, expected_retarget_method="tele_gmr_football"
    )
    == TELE_GMR_MOTION_FILES
  )
  assert len(TELE_GMR_MOTION_FILES) == 43
  assert set(TELE_GMR_MOTION_FILES).isdisjoint(MOTION_FILES)

  baseline = load_env_cfg(STAGE1_TASK)
  tele_gmr = load_env_cfg(TELE_GMR_STAGE1_TASK)
  assert tele_gmr.commands["motion"].motion_files == TELE_GMR_MOTION_FILES
  assert baseline.commands["motion"].motion_files == MOTION_FILES
  assert (
    tele_gmr.commands["motion"].motion_files != baseline.commands["motion"].motion_files
  )
  baseline.commands["motion"].motion_files = TELE_GMR_MOTION_FILES
  # Both factories use the same Stage-1 constructor and robot specialization.
  # Terrain carries a freshly-created callable, so compare the actual training
  # configuration fields rather than callable object identity.
  for field in (
    "decimation",
    "observations",
    "actions",
    "events",
    "sim",
    "episode_length_s",
    "rewards",
    "terminations",
    "commands",
    "scale_rewards_by_dt",
    "scale_rewards_by_motion_dt",
  ):
    assert getattr(tele_gmr, field) == getattr(baseline, field)
  assert tele_gmr.scene.num_envs == baseline.scene.num_envs
  assert tele_gmr.scene.entities == baseline.scene.entities
  assert load_rl_cfg(TELE_GMR_STAGE1_TASK) == load_rl_cfg(STAGE1_TASK)


def test_g1_stage2_keeps_tracker_and_ball_contracts_separate():
  cfg = load_env_cfg(STAGE2_TASK)
  action = cfg.actions["joint_pos"]
  ball = cfg.commands["ball"]

  assert isinstance(action, FrozenTrackerActionCfg)
  assert action.actuator_names == G1_PINGPONG_JOINT_NAMES
  assert action.tracker_file == "ckpts/g1_pingpong/tracker.pt"
  assert action.hit_arm_scale == 1.0
  assert isinstance(ball, ReceivingBallCommandCfg)
  assert ball.racket_site_name == G1_PINGPONG_RACKET_SITE
  assert ball.racket_body_name == "right_wrist_yaw_link"
  assert ball.racket_offset == (0.215, 0.003, 0.0)
  assert ball.racket_normal_axis == 1
  assert ball.incoming_vertical_speed == -3.5
  assert not cfg.commands["motion"].random_dt_training_enabled
  assert load_rl_cfg(STAGE2_TASK).experiment_name == "g1_pingpong_receive_stage2"


def test_g1_tele_gmr_stage2_uses_matching_motion_and_tracker():
  baseline = load_env_cfg(STAGE2_TASK)
  tele_gmr = load_env_cfg(TELE_GMR_STAGE2_TASK)
  assert tele_gmr.commands["motion"].motion_files == TELE_GMR_MOTION_FILES
  assert tele_gmr.actions["joint_pos"].tracker_file == TELE_GMR_TRACKER
  assert baseline.actions["joint_pos"].tracker_file != TELE_GMR_TRACKER
  assert tele_gmr.actions["joint_pos"].actuator_names == G1_PINGPONG_JOINT_NAMES
  assert tele_gmr.commands["ball"] == baseline.commands["ball"]
  assert tele_gmr.rewards == baseline.rewards
  assert tele_gmr.terminations == baseline.terminations
  assert load_rl_cfg(TELE_GMR_STAGE2_TASK).experiment_name == (
    "g1_pingpong_receive_stage2_tele_gmr"
  )
  assert (
    load_rl_cfg(TELE_GMR_STAGE2_TASK).algorithm == load_rl_cfg(STAGE2_TASK).algorithm
  )


def test_g1_play_uses_full_local_dataset_without_training_noise():
  stage1 = load_env_cfg(STAGE1_TASK, play=True)
  stage2 = load_env_cfg(STAGE2_TASK, play=True)
  assert stage1.scene.num_envs == stage2.scene.num_envs == 1
  assert stage1.events == stage2.events == {}
  assert stage1.commands["motion"].sampling_mode == "start"
  assert all(not group.enable_corruption for group in stage2.observations.values())


def test_g1_tele_gmr_hard_stage2_expands_ball_and_speed_distribution():
  baseline = load_env_cfg(TELE_GMR_STAGE2_TASK)
  hard = load_env_cfg(TELE_GMR_HARD_STAGE2_TASK)
  hard_ball = hard.commands["ball"]
  base_ball = baseline.commands["ball"]
  assert hard_ball.intercept_jitter == (0.12, 0.25, 0.08)
  assert hard_ball.target_jitter == (0.30, 0.45)
  assert hard_ball.arrival_time_jitter == 0.18
  assert hard_ball.flight_time_range == (0.30, 0.70)
  assert hard_ball.incoming_speed_range == (2.5, 5.5)
  assert hard_ball.incoming_lateral_speed_range == (-1.5, 1.5)
  assert hard_ball.difficulty_ramp_start_iteration == 30000
  assert hard_ball.difficulty_ramp_steps == 5000
  assert base_ball.incoming_lateral_speed_range == (0.0, 0.0)
  assert hard.actions["joint_pos"].min_speed == 0.35
  assert hard.actions["joint_pos"].max_speed == 2.5
  assert hard.commands["motion"].motion_files == TELE_GMR_MOTION_FILES
  assert hard.actions["joint_pos"].tracker_file == TELE_GMR_TRACKER

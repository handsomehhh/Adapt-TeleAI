"""Contracts that keep the G1 tracker independent of the original tennis task."""

import ast
from pathlib import Path

import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)
from mjlab.tasks.g1_stage1 import TASK_ID
from mjlab.tasks.g1_stage1.env_cfg import g1_stage1_env_cfg
from mjlab.tasks.g1_stage1.mdp.tracking import (
  Stage1MotionCommand,
  Stage1MotionCommandCfg,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg


def test_tracker_task_and_fresh_initialization():
  assert TASK_ID in list_tasks()
  assert not load_rl_cfg(TASK_ID).resume
  cfg = g1_stage1_env_cfg()
  assert tuple(cfg.commands) == ("motion",)
  assert cfg.commands["motion"].joint_names == G1_PINGPONG_JOINT_NAMES
  assert cfg.actions["joint_pos"].actuator_names == G1_PINGPONG_JOINT_NAMES
  assert cfg.commands["motion"].random_dt_training_enabled
  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
  assert cfg.commands["motion"].dt_delta_range == (-0.01, 0.02)
  assert not cfg.scale_rewards_by_motion_dt
  assert not any("Stage2" in task or "Receive-G1" in task for task in list_tasks())


def test_g1_modules_have_no_tennis_or_removed_task_imports():
  root = Path(__file__).parents[1] / "src/mjlab/tasks/g1_stage1"
  forbidden = ("mjlab.tasks.adapt_tennis", "mjlab.tasks.a3_pingpong", "deploy")
  for path in root.rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      if isinstance(node, ast.ImportFrom):
        assert not (node.module or "").startswith(forbidden), path
      elif isinstance(node, ast.Import):
        assert all(not item.name.startswith(forbidden) for item in node.names), path


def test_speed_changes_reference_time_without_changing_physical_discount():
  motion = object.__new__(Stage1MotionCommand)
  motion.cfg = Stage1MotionCommandCfg(
    entity_name="robot",
    anchor_body_name="torso_link",
    body_names=("torso_link",),
    resampling_time_range=(1e9, 1e9),
  )
  motion.last_dt = torch.tensor([-0.01, 0.0, 0.02])
  torch.testing.assert_close(motion.reward_timeline_scale(), torch.ones(3))


def test_play_disables_random_reference_speed_and_noise():
  cfg = g1_stage1_env_cfg(play=True)
  assert not cfg.commands["motion"].random_dt_training_enabled
  assert cfg.commands["motion"].sampling_mode == "start"
  assert cfg.events == {}
  assert all(not group.enable_corruption for group in cfg.observations.values())

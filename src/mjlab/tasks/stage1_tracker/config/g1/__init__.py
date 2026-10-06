"""Unitree G1 generic stage-1 tracker registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_stage1_tracker_env_cfg
from .rl_cfg import unitree_g1_stage1_tracker_ppo_runner_cfg

TASK_ID = "Mjlab-Stage1-Tracker-Flat-Unitree-G1"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_stage1_tracker_env_cfg(),
  play_env_cfg=unitree_g1_stage1_tracker_env_cfg(play=True),
  rl_cfg=unitree_g1_stage1_tracker_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

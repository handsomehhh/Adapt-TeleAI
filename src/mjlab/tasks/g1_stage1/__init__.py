"""Independent G1 ping-pong stage-1 tracker task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import g1_stage1_env_cfg
from .rl_cfg import g1_stage1_runner_cfg
from .runner import G1Stage1Runner

TASK_ID = "Mjlab-G1-PingPong-Stage1"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=g1_stage1_env_cfg(),
  play_env_cfg=g1_stage1_env_cfg(play=True),
  rl_cfg=g1_stage1_runner_cfg(),
  runner_cls=G1Stage1Runner,
)

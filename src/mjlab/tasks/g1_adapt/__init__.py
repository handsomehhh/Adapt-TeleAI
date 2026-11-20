"""AdaPT-style G1 rally and idle planning, isolated from original tennis."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import g1_adapt_env_cfg, g1_adapt_runner_cfg
from .runner import AdaPTRallyRunner

RALLY_TASK = "Mjlab-G1-AdaPT-Rally"
STAND_TASK = "Mjlab-G1-AdaPT-Standing"

for task, standing in ((RALLY_TASK, False), (STAND_TASK, True)):
  register_mjlab_task(
    task_id=task,
    env_cfg=g1_adapt_env_cfg(standing=standing),
    play_env_cfg=g1_adapt_env_cfg(play=True, standing=standing),
    rl_cfg=g1_adapt_runner_cfg(standing),
    runner_cls=AdaPTRallyRunner,
  )

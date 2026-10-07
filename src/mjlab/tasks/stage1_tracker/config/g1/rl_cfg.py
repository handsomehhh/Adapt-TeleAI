"""PPO settings for the generic G1 stage-1 tracker."""

from dataclasses import replace

from mjlab.tasks.adapt_tennis.config.g1.rl_cfg import (
  unitree_g1_adapt_tennis_ppo_runner_cfg,
)


def unitree_g1_stage1_tracker_ppo_runner_cfg():
  """Return the proven stage-1 PPO shape with a tracker-specific run name."""
  cfg = unitree_g1_adapt_tennis_ppo_runner_cfg()
  return replace(
    cfg,
    experiment_name="g1_stage1_tracker",
    save_interval=500,
    max_iterations=10_000,
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    # The 29-DoF G1 action scales are larger than A3's. A full-unit Gaussian
    # at initialization creates implausible joint targets and terminates most
    # environments before PPO sees a useful multi-step trajectory.
    actor=replace(
      cfg.actor,
      distribution_cfg={
        **(cfg.actor.distribution_cfg or {}),
        "init_std": 0.2,
      },
    ),
  )

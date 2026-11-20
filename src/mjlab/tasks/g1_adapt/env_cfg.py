"""Full rally/standing tasks: learned z + speed -> MVAE -> frozen tracker."""

import os
from copy import deepcopy
from dataclasses import fields
from typing import cast

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.g1_stage1.actions import OrderedJointPositionActionCfg
from mjlab.tasks.g1_stage1.env_cfg import g1_stage1_env_cfg
from mjlab.tasks.g1_stage1.rl_cfg import g1_stage1_runner_cfg
from mjlab.tasks.g1_stage1.training_profile import training_profile

from .actions import LatentTrackerActionCfg, tracker_action_history
from .ball import RallyBallCfg
from .reference import GeneratedMotionCommandCfg, generated_history
from .terms import (
  ball_finished,
  ball_observation,
  ball_truth_critic,
  fallen,
  task_reward,
)


def table_spec():
  return mujoco.MjSpec.from_string("""<mujoco><worldbody>
    <body name="table_top" pos="2.02 0 0.74"><geom type="box" size="1.37 0.7625 0.02" rgba="0.08 0.25 0.5 1"/></body>
    <body name="net" pos="2.02 0 0.83625"><geom type="box" size="0.002 0.7625 0.07625" rgba="0.7 0.7 0.7 1"/></body>
  </worldbody></mujoco>""")


def g1_adapt_env_cfg(play=False, standing=False):
  cfg = g1_stage1_env_cfg(play)
  old = cfg.commands["motion"]
  motion = GeneratedMotionCommandCfg(
    **{f.name: getattr(old, f.name) for f in fields(old)}
  )
  motion.mvae_file = os.environ.get("ADAPT_G1_MVAE", "artifacts/g1_adapt/mvae.pt")
  motion.random_dt_training_enabled = False
  motion.sampling_mode = "start"
  motion.debug_vis = False
  cfg.commands["motion"] = motion
  cfg.commands["ball"] = RallyBallCfg(
    resampling_time_range=(1e9, 1e9),
    active=not standing,
    noise=0.0 if play else 0.005,
    dropout=0.0 if play else 0.05,
  )
  cfg.scene.entities["table"] = EntityCfg(spec_fn=table_spec)
  cfg.observations["tracker"] = deepcopy(cfg.observations["actor"])
  cfg.observations["tracker"].terms["actions"] = ObservationTermCfg(
    func=tracker_action_history
  )
  cfg.observations["tracker"].enable_corruption = False
  for name in ("actor", "critic"):
    cfg.observations[name].terms["generated_history"] = ObservationTermCfg(
      func=generated_history
    )
    cfg.observations[name].terms["ball"] = ObservationTermCfg(func=ball_observation)
  cfg.observations["critic"].terms["ball_truth"] = ObservationTermCfg(
    func=ball_truth_critic
  )
  old_action = cast(OrderedJointPositionActionCfg, cfg.actions["joint_pos"])
  cfg.actions["joint_pos"] = LatentTrackerActionCfg(
    entity_name="robot",
    actuator_names=old_action.actuator_names,
    scale=old_action.scale,
    preserve_order=True,
    use_default_offset=True,
    tracker_file=os.environ.get("ADAPT_G1_TRACKER", "artifacts/g1_adapt/tracker.pt"),
  )
  profile = training_profile().get("planner", {})
  for name in ("min_speed", "max_speed", "latent_clip"):
    if name in profile:
      setattr(cfg.actions["joint_pos"], name, profile[name])
  cfg.rewards = {}
  weights = (
    {
      "standing": 3.0,
      "quiet": -0.02,
      "latent": -0.01,
      "speed": -0.01,
      "tilt": -2.0,
      "backward": -1.0,
    }
    if standing
    else {
      "approach": 1.0,
      "hit": 5.0,
      "net": 30.0,
      "landing": 100.0,
      "forward": 5.0,
      "double_hit": -10.0,
      "latent": -0.01,
      "speed": -0.01,
      "tilt": -2.0,
      "backward": -1.0,
    }
  )
  weights.update(profile.get("standing_weights" if standing else "rally_weights", {}))
  for kind, weight in weights.items():
    cfg.rewards[kind] = RewardTermCfg(
      func=task_reward,
      weight=weight,
      params={"kind": kind, "event_rewards": profile.get("event_rewards", False)},
    )
  cfg.terminations = {"fallen": TerminationTermCfg(func=fallen)}
  cfg.terminations["time_out"] = TerminationTermCfg(func=time_out, time_out=True)
  if not standing:
    cfg.terminations["ball_finished"] = TerminationTermCfg(func=ball_finished)
  cfg.episode_length_s = 4.0 if not standing else 8.0
  if play:
    cfg.episode_length_s = 1e9
  cfg.viewer.lookat = (1.2, 0.0, 0.8)
  cfg.viewer.distance = 4.5
  return cfg


def g1_adapt_runner_cfg(standing=False):
  cfg = g1_stage1_runner_cfg()
  cfg.experiment_name = "g1_adapt_standing" if standing else "g1_adapt_rally"
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.5
  cfg.algorithm.entropy_coef = 0.0
  return cfg

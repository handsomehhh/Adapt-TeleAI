"""Planner observations, task rewards and result terminations."""

import torch

from mjlab.utils.lab_api.math import matrix_from_quat


def ball_observation(env):
  return env.command_manager.get_term("ball").command


def ball_truth_critic(env):
  ball = env.command_manager.get_term("ball")
  return torch.cat((ball.p - env.scene.env_origins, ball.v), -1)


def ball_finished(env):
  return env.command_manager.get_term("ball").done


def fallen(env):
  robot = env.scene["robot"]
  return (robot.data.root_link_pos_w[:, 2] < 0.45) | (
    robot.data.projected_gravity_b[:, 2] > -0.5
  )


def task_reward(env, kind: str, event_rewards: bool = False):
  ball = env.command_manager.get_term("ball")
  motion = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  if kind == "approach":
    paddle, _ = ball.paddle()
    phase = ((1 - motion.generator.state[:, 70]) * 0.5).clamp(0, 1)
    return torch.exp(-(paddle - ball.p).square().sum(-1) / 0.04) * phase * ~ball.hit
  if kind == "hit":
    return (ball.hit_event if event_rewards else ball.hit).float()
  if kind == "net":
    return (ball.net_event if event_rewards else ball.net_clear).float()
  if kind == "landing":
    return ball.landing_score
  if kind == "forward":
    return ball.hit * ball.v[:, 0].clamp(0, 8) / 8
  if kind == "double_hit":
    return ball.double_hit.float()
  if kind == "latent":
    return env.action_manager.action[:, :-1].square().mean(-1)
  if kind == "speed":
    return (env.action_manager.get_term("joint_pos").speed - 1).square()
  if kind == "tilt":
    return robot.data.projected_gravity_b[:, :2].square().sum(-1)
  if kind == "backward":
    return (-matrix_from_quat(robot.data.root_link_quat_w)[:, 0, 0]).clamp_min(0)
  if kind == "standing":
    qids = motion._motion_to_robot_joint_ids
    qerror = (robot.data.joint_pos[:, qids] - motion.ready_joints).square().mean(-1)
    perror = (robot.data.root_link_pos_w - motion.ready_root).square().sum(-1)
    return torch.exp(-qerror / 0.1 - perror / 0.04)
  if kind == "quiet":
    return robot.data.joint_vel.square().mean(-1)
  if kind == "tracking":
    qerror = (
      (robot.data.joint_pos[:, motion._motion_to_robot_joint_ids] - motion.joint_pos)
      .square()
      .mean(-1)
    )
    return torch.exp(-qerror / 0.04)
  if kind == "fall":
    return fallen(env).float()
  raise ValueError(kind)

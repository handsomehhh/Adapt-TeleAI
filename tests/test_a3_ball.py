"""CPU tests of physical events; no simulator or policy is needed."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.tasks.a3_pingpong.ball import (
  ReceivingBallCommand,
  ReceivingBallCommandCfg,
  ball_intercept_reward,
  ballistic_step,
  swept_net_collision,
  swept_racket_collision,
  swept_table_collision,
)
from mjlab.tasks.adapt_tennis.mdp.commands import _batched_quat_slerp


def tensor(values):
  return torch.tensor(values, dtype=torch.float32)


def test_swept_paddle_catches_tunnelling_and_rejects_near_miss():
  start = tensor([[0.2, 0, 1], [0.2, 0.09, 1]])
  end = tensor([[-0.2, 0, 1], [-0.2, 0.09, 1]])
  center = tensor([[0, 0, 1], [0, 0, 1]])
  normal = tensor([[1, 0, 0], [1, 0, 0]])
  position, velocity, hit = swept_racket_collision(
    start,
    end,
    tensor([[-20, 0, 0], [-20, 0, 0]]),
    center,
    center,
    normal,
    0.02,
    restitution=1,
  )
  assert hit.tolist() == [True, False]
  assert velocity[0, 0] == 20
  assert position[0, 0] > 0.02
  torch.testing.assert_close(position[1], end[1])


def test_moving_paddle_transfers_velocity_and_both_faces_work():
  start = tensor([[0.08, 0, 1], [-0.08, 0, 1]])
  end = tensor([[0.04, 0, 1], [-0.04, 0, 1]])
  old_center = tensor([[0, 0, 1], [0, 0, 1]])
  center = tensor([[0.04, 0, 1], [-0.04, 0, 1]])
  normal = tensor([[1, 0, 0], [1, 0, 0]])
  _, velocity, hit = swept_racket_collision(
    start,
    end,
    tensor([[-2, 0, 0], [2, 0, 0]]),
    old_center,
    center,
    normal,
    0.02,
    restitution=1,
  )
  assert hit.all()
  torch.testing.assert_close(velocity[:, 0], tensor([6, -6]))


def test_ball_moving_away_does_not_hit():
  center = tensor([[0, 0, 1]])
  _, _, hit = swept_racket_collision(
    tensor([[0.01, 0, 1]]),
    tensor([[0.05, 0, 1]]),
    tensor([[2, 0, 0]]),
    center,
    center,
    tensor([[1, 0, 0]]),
    0.02,
  )
  assert not hit.any()


def test_table_bounce_requires_real_downward_crossing_inside_table():
  start = tensor([[2.5, 0, 0.9], [2.5, 1, 0.9], [0.4, 0, 0.9]])
  end = start.clone()
  end[:, 2] = 0.7
  _, velocity, hit, landing = swept_table_collision(
    start, end, tensor([[1, 0, -10]] * 3), 0.02
  )
  assert hit.tolist() == [True, False, False]
  assert velocity[0, 2] > 0
  assert abs(float(landing[0, 2]) - 0.78) < 1e-6


def test_net_blocks_low_shot_and_counts_actual_clearance_only():
  start = tensor([[1.9, 0, 0.85], [1.9, 0, 1.0], [1.9, 1, 1.0], [2.2, 0, 1.0]])
  end = tensor([[2.2, 0, 0.85], [2.2, 0, 1.0], [2.2, 1, 1.0], [1.9, 0, 1.0]])
  _, velocity, blocked, cleared = swept_net_collision(
    start, end, tensor([[15, 0, 0], [15, 0, 0], [15, 0, 0], [-15, 0, 0]]), 0.02
  )
  assert blocked.tolist() == [True, False, False, False]
  assert cleared.tolist() == [False, True, False, False]
  assert velocity[0, 0] < 0


def test_ballistic_feed_intercepts_without_position_correction():
  target = tensor([[0.6, -0.2, 1.1]])
  duration = 0.4
  final_velocity = tensor([[-4, 0, -2.7]])
  velocity = final_velocity.clone()
  velocity[:, 2] += 9.81 * duration
  position = target - velocity * duration
  position[:, 2] += 0.5 * 9.81 * duration**2
  for _ in range(80):
    position, velocity = ballistic_step(position, velocity, 0.005)
  torch.testing.assert_close(position, target, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(velocity, final_velocity, atol=1e-5, rtol=1e-5)


def reference_ball_fixture(arrival_time_jitter=0.0):
  """Use actual clips with kinematic paddles to separate feeder from policy errors."""
  paths = sorted(
    (Path(__file__).parents[1] / "dataset/a3_pingpong/motions").glob("*.npz")
  )
  if not paths:
    pytest.skip("Optional A3 reference dataset is not installed")
  arrays = []
  for path in paths:
    with np.load(path) as data:
      wrist_id = list(data["body_names"]).index("right_wrist_yaw_Link")
      arrays.append(
        (
          data["racket_pos_w"],
          data["body_quat_w"][:, wrist_id],
          data["body_pos_w"][:, wrist_id],
        )
      )
  racket, quaternion, wrist = [
    tensor(np.stack([a[i] for a in arrays])) for i in range(3)
  ]
  count = len(paths)
  data = SimpleNamespace(
    site_pos_w=racket[:, 0, None].clone(),
    body_link_quat_w=quaternion[:, 0, None].clone(),
    root_link_pos_w=torch.zeros(count, 3),
    root_link_quat_w=tensor([[1, 0, 0, 0]]).repeat(count, 1),
  )
  robot = SimpleNamespace(
    data=data,
    body_names=["right_wrist_yaw_Link"],
    find_sites=lambda *args, **kwargs: ([0], ["racket_center"]),
  )

  class Scene(dict):
    pass

  scene = Scene(robot=robot)
  scene.env_origins = torch.zeros(count, 3)
  motion = SimpleNamespace(
    motion_ids=torch.arange(count),
    cfg=SimpleNamespace(body_names=("right_wrist_yaw_Link",)),
    current_time_seconds=lambda ids: torch.zeros(len(ids)),
    body_pos_w=wrist[:, 0, None],
    body_quat_w=quaternion[:, 0, None],
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    scene=scene,
    command_manager=SimpleNamespace(get_term=lambda name: motion),
  )
  cfg = ReceivingBallCommandCfg(
    resampling_time_range=(1e9, 1e9),
    intercept_jitter=(0, 0, 0),
    target_jitter=(0, 0),
    flight_time_range=(0.45, 0.45),
    incoming_speed_range=(4, 4),
    position_noise=0,
    velocity_noise=0,
    arrival_time_jitter=arrival_time_jitter,
  )
  ball = ReceivingBallCommand(cfg, env)
  ball._reference_hit_pos = racket[:, 43].clone()
  ball._reference_hit_time = torch.full((count,), 0.86)
  ball._resample_command(torch.arange(count))
  env.command_manager.get_term = lambda name: ball if name == "ball" else motion

  def gather_reference(attribute, motion_ids, times):
    frames = (times * 50).clamp(0, wrist.shape[1] - 1)
    first = frames.long()
    second = (first + 1).clamp_max(wrist.shape[1] - 1)
    blend = frames - first
    if attribute == "body_quat_w":
      value = _batched_quat_slerp(
        quaternion[motion_ids, first], quaternion[motion_ids, second], blend
      )
    else:
      value = (1 - blend[:, None]) * wrist[motion_ids, first] + blend[:, None] * wrist[
        motion_ids, second
      ]
    return value[:, None]

  motion.motion = SimpleNamespace(_gather_interp=gather_reference)
  return ball, data, racket, quaternion


def test_reference_paddles_can_hit_feeds_but_success_requires_return():
  ball, data, racket, quaternion = reference_ball_fixture()
  assert (ball.pos_w[:, 2] > ball.cfg.table_height + ball.cfg.ball_radius).all()
  for step in range(600):
    frame = min((step + 1) * 0.005 * 50, racket.shape[1] - 1)
    first, second = int(frame), min(int(frame) + 1, racket.shape[1] - 1)
    fraction = frame - first
    data.site_pos_w[:, 0] = (1 - fraction) * racket[:, first] + fraction * racket[
      :, second
    ]
    data.body_link_quat_w[:, 0] = _batched_quat_slerp(
      quaternion[:, first],
      quaternion[:, second],
      torch.full((ball.num_envs,), fraction),
    )
    ball.compute_substep(0.005)
  assert ball.hit.all(), "Feeder must not make the supplied reference swings impossible"
  assert not ball.success.all(), "Contact alone must never imply legal return"
  assert torch.all(
    ~ball.success | (ball.hit & ball.landed & ball.net_cleared & ~ball.net_hit)
  )
  assert ball.command.shape == (ball.num_envs, 23)
  assert torch.isfinite(ball.command).all()


def test_landing_reward_event_pays_once_and_latch_survives():
  ball, _, _, _ = reference_ball_fixture()
  ball.launch_time[:] = 0
  ball.pos_w[:] = tensor([2.7, 0, 0.785])
  ball.vel_w[:] = tensor([0.5, 0, -1.0])
  ball.hit[:] = True
  ball.net_cleared[:] = True
  ball.compute_substep(0.01)
  assert ball.success.all() and ball.landing_event.all()
  ball._update_command()
  ball.compute_substep(0.01)
  assert ball.success.all() and not ball.landing_event.any()


def test_optional_arrival_jitter_changes_timing_without_snapping_ball():
  ball, _, _, _ = reference_ball_fixture(arrival_time_jitter=0.08)
  assert (ball.intercept_time >= 0.78).all()
  assert (ball.intercept_time <= 0.94).all()
  assert ball.intercept_time.std() > 0.01
  duration = ball.intercept_time - ball.launch_time
  predicted, _ = ballistic_step(ball.pos_w, ball.vel_w, duration)
  torch.testing.assert_close(predicted, ball.intercept_w, atol=1e-6, rtol=1e-5)


def test_intercept_reward_follows_ball_schedule_not_policy_phase():
  ball, data, _, _ = reference_ball_fixture(arrival_time_jitter=0.08)
  motion = ball._env.command_manager.get_term("motion")
  ball.elapsed[:] = ball.intercept_time
  data.site_pos_w[:, 0] = ball.intercept_w
  on_time = ball_intercept_reward(ball._env)
  torch.testing.assert_close(on_time, torch.ones_like(on_time), atol=1e-5, rtol=1e-5)
  # Artificially rewind the learned reference while holding all physical state
  # fixed. Rewinding must not change the desired interception pose or reward.
  motion.body_pos_w = torch.full_like(motion.body_pos_w, 100.0)
  motion._time_seconds_progress = torch.zeros(ball.num_envs)
  rewound = ball_intercept_reward(ball._env)
  torch.testing.assert_close(rewound, on_time)
  # Remaining at an early pose when the ball arrives receives less guidance.
  data.site_pos_w[:, 0, 0] += 0.3
  late = ball_intercept_reward(ball._env)
  assert (late < 0.03).all()

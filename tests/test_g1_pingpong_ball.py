"""Reference-feeder checks for the 29-DoF G1 ping-pong motions."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_WRIST_BODY,
)
from mjlab.tasks.a3_pingpong import ball as ball_module
from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommand, ReceivingBallCommandCfg
from mjlab.tasks.adapt_tennis.mdp.commands import _batched_quat_slerp

PHYSICS_DT = 0.005
MOTION_FPS = 50
STRIKE_FRAME = 43
STRIKE_TIME_S = STRIKE_FRAME / MOTION_FPS


@pytest.fixture(scope="module")
def g1_reference_swings():
  paths = sorted(
    (Path(__file__).parents[1] / "dataset/g1_pingpong/motions").glob("*.npz")
  )
  if len(paths) != 43:
    pytest.skip("Optional G1 reference dataset is not installed.")

  arrays = []
  for path in paths:
    with np.load(path, allow_pickle=False) as data:
      wrist_id = list(data["body_names"]).index(G1_PINGPONG_WRIST_BODY)
      assert int(data["strike_frame"]) == STRIKE_FRAME
      arrays.append(
        (
          data["racket_pos_w"],
          data["body_quat_w"][:, wrist_id],
          data["body_pos_w"][:, wrist_id],
        )
      )

  racket, quaternion, wrist = (
    torch.tensor(np.stack([clip[index] for clip in arrays]), dtype=torch.float32)
    for index in range(3)
  )
  return paths, racket, quaternion, wrist


def _failed_names(paths: list[Path], mask: torch.Tensor) -> list[str]:
  return [
    path.name for path, failed in zip(paths, mask.tolist(), strict=True) if failed
  ]


def _run_reference_feeds(
  g1_reference_swings, flight_time: float, incoming_speed: float
):
  paths, racket, quaternion, wrist = g1_reference_swings
  count = len(paths)
  robot_data = SimpleNamespace(
    site_pos_w=racket[:, 0, None].clone(),
    body_link_quat_w=quaternion[:, 0, None].clone(),
    root_link_pos_w=torch.zeros(count, 3),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
  )
  robot = SimpleNamespace(
    data=robot_data,
    body_names=[G1_PINGPONG_WRIST_BODY],
    find_sites=lambda *args, **kwargs: ([0], [G1_PINGPONG_RACKET_SITE]),
  )

  class Scene(dict):
    pass

  scene = Scene(robot=robot)
  scene.env_origins = torch.zeros(count, 3)
  motion = SimpleNamespace(
    motion_ids=torch.arange(count),
    cfg=SimpleNamespace(body_names=(G1_PINGPONG_WRIST_BODY,)),
    current_time_seconds=lambda env_ids: torch.zeros(len(env_ids)),
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
    racket_site_name=G1_PINGPONG_RACKET_SITE,
    racket_body_name=G1_PINGPONG_WRIST_BODY,
    racket_offset=G1_PINGPONG_MOUNT_OFFSET,
    intercept_jitter=(0.0, 0.0, 0.0),
    target_jitter=(0.0, 0.0),
    flight_time_range=(flight_time, flight_time),
    incoming_speed_range=(incoming_speed, incoming_speed),
    incoming_vertical_speed=-3.5,
    position_noise=0.0,
    velocity_noise=0.0,
    arrival_time_jitter=0.0,
  )
  ball = ReceivingBallCommand(cfg, env)
  ball._reference_hit_pos = racket[:, STRIKE_FRAME].clone()
  ball._reference_hit_time = torch.full((count,), STRIKE_TIME_S)
  ball._resample_command(torch.arange(count))

  incoming_net_contact = torch.zeros(count, dtype=torch.bool)
  incoming_table_contact = torch.zeros(count, dtype=torch.bool)
  original_net_collision = ball_module.swept_net_collision
  original_table_collision = ball_module.swept_table_collision

  def record_net_collision(*args, **kwargs):
    result = original_net_collision(*args, **kwargs)
    incoming_net_contact.logical_or_(result[2] & ~ball.hit)
    return result

  def record_table_collision(*args, **kwargs):
    result = original_table_collision(*args, **kwargs)
    incoming_table_contact.logical_or_(result[2] & ~ball.hit)
    return result

  assert (ball.pos_w[:, 2] > cfg.table_height + cfg.ball_radius).all()
  with (
    patch.object(ball_module, "swept_net_collision", record_net_collision),
    patch.object(ball_module, "swept_table_collision", record_table_collision),
  ):
    for step in range(240):
      frame = min((step + 1) * PHYSICS_DT * MOTION_FPS, racket.shape[1] - 1)
      first = int(frame)
      second = min(first + 1, racket.shape[1] - 1)
      fraction = frame - first
      robot_data.site_pos_w[:, 0] = (1.0 - fraction) * racket[
        :, first
      ] + fraction * racket[:, second]
      robot_data.body_link_quat_w[:, 0] = _batched_quat_slerp(
        quaternion[:, first],
        quaternion[:, second],
        torch.full((count,), fraction),
      )
      ball.compute_substep(PHYSICS_DT)

  return paths, ball, incoming_net_contact, incoming_table_contact


@pytest.mark.parametrize("flight_time", (0.40, 0.45, 0.50))
@pytest.mark.parametrize("incoming_speed", (3.5, 4.0, 4.5))
def test_all_g1_reference_swings_reach_clean_incoming_feeds(
  g1_reference_swings, flight_time, incoming_speed
):
  paths, ball, incoming_net_contact, incoming_table_contact = _run_reference_feeds(
    g1_reference_swings, flight_time, incoming_speed
  )

  assert not incoming_net_contact.any(), (
    "Incoming feed hit the net before the reference racket: "
    f"{_failed_names(paths, incoming_net_contact)}"
  )
  assert not incoming_table_contact.any(), (
    "Incoming feed hit the table before the reference racket: "
    f"{_failed_names(paths, incoming_table_contact)}"
  )
  assert ball.hit.all(), (
    "Feeder made supplied G1 reference swings impossible: "
    f"{_failed_names(paths, ~ball.hit)}"
  )

"""Real ROS command decoding and hitter selection, without robot IO."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from deploy.hope_g1.controller import HopeController
from deploy.hope_g1_hitter.configuration import load_deployment
from deploy.hope_g1_hitter.controller import HitterController
from deploy.racket_command import RacketCommand


@pytest.fixture(scope="module")
def deployment():
  torch.set_num_threads(1)
  directory = Path(__file__).resolve().parents[1] / "hope_g1_hitter"
  options = yaml.safe_load((directory / "configs/rally.yaml").read_text())
  runtime = yaml.safe_load((directory / "configs/real.yaml").read_text())
  model = load_deployment(
    directory / "artifacts/hitter_20261027_9500",
    ball_config=options["ball"],
    world_config=options["world"],
  )
  return model, runtime, options


@pytest.fixture
def hitter(deployment):
  return HitterController(*deployment)


def packet_for(hitter):
  return {
    "valid": True,
    "position": hitter.geometry.centre(1, hitter.station).tolist(),
    "velocity": hitter.geometry.velocity_boxes[1].mean(axis=1).tolist(),
    "normal": [1.0, 0.0, 0.0],
    "time_to_strike": 0.6,
    "bridge_monotonic": 100.0,
    "stamp": 1000.1,
    "source_ball_time": 1000.0,
    "outgoing_velocity": [2.0, 0.0, 1.0],
    "clears_net": True,
    "bypasses_net_posts": False,
    "predicted_bounces": 1,
  }


def test_first_command_enters_swing_and_runs_actor(hitter):
  packet = packet_for(hitter)
  hitter.update_external_planner_command(packet, 100.02)
  command = hitter.external_planner_command
  assert isinstance(command, RacketCommand)
  assert command.diagnostic["valid"] is True
  assert command.diagnostic["source_stamp"] == 1000.0
  assert command.diagnostic["num_bounces"] == 1
  assert command.deadline == pytest.approx(100.6)
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.02)
  state = SimpleNamespace(
    q=hitter.controller.default.copy(),
    dq=np.zeros(29),
    gyro=np.zeros(3),
    root_pos=np.r_[hitter.station, 0.0],
    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
  )
  target = hitter.step(state, None, 100.03)
  assert target.shape == (29,) and np.isfinite(target).all()
  assert hitter.controller.debug["actor_obs"].shape == (104,)
  assert hitter.lifecycle.state == "EXECUTE"
  assert hitter.shot.side == 1
  assert hitter.shot.diagnostic["reason"] == "accepted"
  assert hitter.events[-1]["event"] == "motion_selected"


def test_repeated_packet_and_selection_do_not_extend_deadline(hitter):
  packet = packet_for(hitter)
  hitter.update_external_planner_command(packet, 100.02)
  for now in (100.03, 100.2, 100.5):
    # The shared runtime can supply the same cached packet each control tick.
    hitter.update_external_planner_command(packet, 100.02)
    shot, _ = hitter.select(now)
    assert shot.strike_at == pytest.approx(100.6)
    assert hitter.external_planner_deadline == pytest.approx(100.6)
  shot, diagnostics = hitter.select(100.61)
  assert shot is None
  assert diagnostics[0]["reason"] == "time_window"


def test_old_hope_and_hitter_use_same_command_contract(hitter):
  hope = HopeController.__new__(HopeController)
  hope.planner_transport = "ros2"
  packet = packet_for(hitter)
  for bridge_stamp in (100.0, None):
    if bridge_stamp is None:
      packet.pop("bridge_monotonic")
    hope.update_external_planner_command(packet, 100.02)
    hitter.update_external_planner_command(packet, 100.02)
    old, new = hope.external_planner_command, hitter.external_planner_command
    for field in ("position", "velocity", "normal"):
      np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    assert old.diagnostic == new.diagnostic
    expected = (100.02 if bridge_stamp is None else bridge_stamp) + 0.6
    assert old.deadline == new.deadline == pytest.approx(expected)


@pytest.mark.parametrize(
  "override, error",
  [
    ({"valid": False}, "ros_planner_published_invalid_command"),
    ({"position": None}, "malformed_ros_planner_command"),
    ({"velocity": [1.0, 2.0]}, "malformed_ros_planner_command"),
    ({"normal": [0.0, 0.0, 0.0]}, "malformed_ros_planner_command"),
    ({"time_to_strike": float("nan")}, "malformed_ros_planner_command"),
    ({"time_to_strike": -0.1}, "malformed_ros_planner_command"),
    ({"bridge_monotonic": float("inf")}, "malformed_ros_planner_command"),
  ],
)
def test_rejected_packet_clears_previous_command(hitter, override, error):
  packet = packet_for(hitter)
  hitter.update_external_planner_command(packet, 100.02)
  hitter.update_external_planner_command(dict(packet, **override), 100.03)
  assert hitter.external_planner_command is None
  assert hitter.external_planner_deadline is None
  shot, diagnostics = hitter.select(100.04)
  assert shot is None
  assert diagnostics[0]["reason"] == error


def test_height_outside_training_box_is_rejected(hitter):
  packet = packet_for(hitter)
  packet["position"][2] += 1.0
  hitter.update_external_planner_command(packet, 100.02)
  shot, diagnostics = hitter.select(100.03)
  assert shot is None
  assert diagnostics[0]["reason"] == "height_outside_training_box"


def test_no_packet_keeps_idle_and_backhand(hitter):
  hitter.update_external_planner_command(None, None)
  shot, diagnostics = hitter.select(100.0)
  assert shot is None
  assert diagnostics[0]["reason"] == "no_valid_external_command"
  assert hitter.side == 1

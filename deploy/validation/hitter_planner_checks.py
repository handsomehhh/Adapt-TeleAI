"""Real ROS command decoding and hitter selection, without robot IO."""

from copy import deepcopy
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


def state_for(hitter, y=None):
  return SimpleNamespace(
    q=hitter.controller.default.copy(), dq=np.zeros(29), gyro=np.zeros(3),
    root_pos=np.array([hitter.station[0], hitter.station[1] if y is None else y, 0.0]),
    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
  )


def common_packet(hitter, side):
  packet = packet_for(hitter)
  packet["position"] = hitter.geometry.centre(side, hitter.station).tolist()
  packet["position"][0] = -0.219
  packet["position"][2] = 0.0  # Inside both original training height ranges.
  packet["velocity"] = hitter.geometry.velocity_boxes[side].mean(axis=1).tolist()
  return packet


@pytest.mark.parametrize("side", [0, 1])
def test_auto_selects_both_sides_and_runs_real_actor(hitter, side):
  packet = common_packet(hitter, side)
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.0)
  hitter.update_external_planner_command(packet, 100.02)
  target = hitter.step(state_for(hitter), None, 100.03)
  assert hitter.side_selection == "auto"
  assert hitter.shot.side == side
  assert np.isfinite(target).all() and target.shape == (29,)
  np.testing.assert_array_equal(hitter.shot.position, packet["position"])
  assert hitter.shot.strike_at == pytest.approx(100.6)
  assert sum(a["selected"] for a in hitter.debug["decision"]["attempts"]) == 1
  assert hitter.events[-1]["side"] == ("forehand", "backhand")[side]


def test_selection_uses_measured_position(hitter):
  packet = common_packet(hitter, 0)
  hitter.update_external_planner_command(packet, 100.02)
  near_home, _ = hitter.select(100.03, state_for(hitter))
  near_backhand_station, _ = hitter.select(100.03, state_for(hitter, y=-1.02))
  assert near_home.side == 0
  assert near_backhand_station.side == 1


def test_side_is_locked_until_follow_through_finishes(hitter):
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.0)
  state = state_for(hitter)
  hitter.update_external_planner_command(common_packet(hitter, 0), 100.02)
  hitter.step(state, None, 100.03)
  assert hitter.side == 0
  hitter.update_external_planner_command(common_packet(hitter, 1), 100.04)
  hitter.step(state, None, 100.05)
  assert hitter.side == 0
  assert len(hitter.debug["decision"]["attempts"]) == 1
  # A fresh ball after the forehand follow-through can select backhand.
  packet = common_packet(hitter, 1)
  packet["position"][1] = hitter.geometry.centre(1, state.root_pos[:2])[1]
  packet["bridge_monotonic"] = 102.0
  hitter.update_external_planner_command(packet, 102.01)
  hitter.step(state, None, 102.02)
  assert hitter.side == 1
  assert hitter.completed == 1


@pytest.mark.parametrize("mode, side", [("forehand", 0), ("backhand", 1)])
def test_fixed_side_mode_and_reset(deployment, mode, side):
  model, runtime, options = deployment
  options = deepcopy(options)
  options["hitter"]["side_selection"] = mode
  hitter = HitterController(model, runtime, options)
  hitter.reset_state()
  hitter.update_external_planner_command(common_packet(hitter, 1 - side), 100.02)
  shot, attempts = hitter.select(100.03)
  assert shot.side == side
  assert len(attempts) == 1
  assert hitter.side == side


def test_backhand_plane_is_not_relabelled_as_forehand(hitter):
  packet = common_packet(hitter, 0)
  packet["position"][0] = -0.256
  hitter.update_external_planner_command(packet, 100.02)
  shot, attempts = hitter.select(100.03)
  assert shot.side == 1
  assert next(a for a in attempts if a["side"] == "forehand")["reason"] == "x_outside_training_box"
  assert shot.position[0] == -0.256
  assert shot.strike_at == pytest.approx(100.6)


def test_x_outside_both_boxes_is_rejected(hitter):
  packet = common_packet(hitter, 0)
  packet["position"][0] = 0.0
  hitter.update_external_planner_command(packet, 100.02)
  shot, attempts = hitter.select(100.03)
  assert shot is None
  assert {a["reason"] for a in attempts} == {"x_outside_training_box"}


def test_default_ros_plane_serves_both_sides(hitter):
  path = Path(__file__).resolve().parents[1] / "hope_g1_hitter/configs/hope_planner_ros2.yaml"
  params = yaml.safe_load(path.read_text())["hope_planner_optimized_2"]["ros__parameters"]
  lo, hi = hitter.geometry.common_x_range()
  assert lo <= params["x_hit"] <= hi
  assert params["publish_intercept_x_min"] == params["publish_intercept_x_max"] == params["x_hit"]


def begin_anchored_shot(hitter, side=0):
  anchor = np.array([-0.70, -0.80, 0.03])
  hitter.capture_return_anchor(anchor, 99.0, source="up_button")
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.0)
  state = state_for(hitter)
  state.root_pos = anchor.copy()
  hitter.update_external_planner_command(common_packet(hitter, side), 100.02)
  hitter.step(state, None, 100.03)
  assert hitter.shot.side == side
  finish = hitter.shot.strike_at + hitter.geometry.follow[side] + 0.02
  return anchor, state, finish


@pytest.mark.parametrize("side", [0, 1])
def test_finished_swing_returns_base_and_racket_targets_to_captured_anchor(hitter, side):
  anchor, state, finish = begin_anchored_shot(hitter, side)
  previous_action = hitter.controller.previous_joint.copy()
  state.root_pos += [0.12, 0.30, 0.04]
  target = hitter.step(state, None, finish)
  assert np.isfinite(target).all()
  assert hitter.completed == 1 and hitter.recovered == 0
  assert hitter.shot is None and hitter.lifecycle.state == "ARMED"
  assert hitter.phase == "RETURN" and hitter.receiving
  np.testing.assert_array_equal(hitter.return_anchor, anchor)
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], anchor[:2])
  expected_racket = hitter.geometry.centre(side, anchor[:2])
  np.testing.assert_allclose(hitter.debug["goal_position"], expected_racket)
  obs = hitter.debug["actor_obs"]
  np.testing.assert_allclose(obs[95:97], anchor[:2] - state.root_pos[:2])
  np.testing.assert_allclose(obs[97:100], expected_racket - state.root_pos)
  np.testing.assert_allclose(obs[61:90], previous_action)
  assert obs[103] == pytest.approx(hitter.geometry.windup[side])
  # Waiting for longer must not make the displaced measured base a new target.
  state.root_pos[1] += 0.05
  hitter.step(state, None, finish + 0.1)
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], anchor[:2])
  np.testing.assert_allclose(hitter.debug["goal_position"], expected_racket)


def test_return_is_complete_only_when_measured_base_reaches_anchor(hitter):
  anchor, state, finish = begin_anchored_shot(hitter)
  state.root_pos[1] += 0.2
  hitter.step(state, None, finish)
  assert hitter.recovered == 0
  state.root_pos = anchor + [0.01, -0.02, 0.1]
  hitter.step(state, None, finish + 0.1)
  assert hitter.phase == "IDLE" and hitter.recovered == 1
  hitter.step(state, None, finish + 0.2)
  assert hitter.recovered == 1
  assert sum(e["event"] == "return_anchor_reached" for e in hitter.events) == 1
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], anchor[:2])


def test_new_command_interrupts_return_and_can_change_side(hitter):
  anchor, state, finish = begin_anchored_shot(hitter, side=0)
  state.root_pos[1] += 0.2
  hitter.step(state, None, finish)
  assert hitter.phase == "RETURN"
  packet = common_packet(hitter, 1)
  packet["position"][1] = hitter.geometry.centre(1, state.root_pos[:2])[1]
  packet["bridge_monotonic"] = finish + 0.02
  hitter.update_external_planner_command(packet, finish + 0.03)
  hitter.step(state, None, finish + 0.04)
  assert hitter.shot.side == 1 and hitter.phase == "SWING"
  assert hitter.debug["return_anchor_error_m"] > hitter.return_tolerance
  assert hitter.recovered == 0
  assert any(e["event"] == "return_interrupted_by_command" for e in hitter.events)
  np.testing.assert_array_equal(hitter.shot.position, packet["position"])
  assert hitter.shot.strike_at == pytest.approx(finish + 0.62)
  np.testing.assert_array_equal(hitter.return_anchor, anchor)
  # The second swing returns to exactly the same first-UP anchor.
  hitter.step(state, None, hitter.shot.strike_at + hitter.geometry.follow[1] + 0.02)
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], anchor[:2])
  assert hitter.completed == 2


def test_return_does_not_pretend_robot_has_already_reached_anchor(hitter):
  anchor, state, finish = begin_anchored_shot(hitter)
  state.root_pos[1] = -1.2
  hitter.step(state, None, finish)
  packet = common_packet(hitter, 1)
  packet["position"][1] = -0.4  # Reachable from home, too far from measured Y.
  packet["bridge_monotonic"] = finish + 0.02
  hitter.update_external_planner_command(packet, finish + 0.03)
  hitter.step(state, None, finish + 0.04)
  assert hitter.shot is None and hitter.phase == "RETURN"
  assert {a["reason"] for a in hitter.debug["decision"]["attempts"]} == {"station_outside_workspace"}
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], anchor[:2])


def test_up_anchor_survives_preposition_and_down_up_reentry(hitter):
  anchor = np.array([-0.71, -0.81, 0.07])
  expected = anchor.copy()
  hitter.capture_return_anchor(anchor, 10.0, source="up_button")
  anchor[:] = 99  # The mocap source buffer may be reused.
  hitter.lifecycle.state = "STABLE"
  hitter.capture_return_anchor([-0.65, -0.75, 0.02], 12.0)
  hitter.start_policy(hitter.controller.default, 12.0)
  state = state_for(hitter, y=-0.75)
  hitter.step(state, None, 12.02)
  hitter.reset_manual_policy(13.0)
  hitter.lifecycle.state = "STABLE"
  hitter.capture_return_anchor([-0.60, -0.60, 0.02], 14.0, source="up_button")
  hitter.start_policy(hitter.controller.default, 16.0)
  hitter.step(state, None, 16.02)
  np.testing.assert_array_equal(hitter.return_anchor, expected)
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], expected[:2])
  assert sum(e["event"] == "return_anchor_captured" for e in hitter.events) == 1


@pytest.mark.parametrize("position", [None, [0.0, 0.0], [0.0, float("nan"), 0.0]])
def test_invalid_first_anchor_is_rejected(hitter, position):
  with pytest.raises(ValueError, match="valid measured base XYZ"):
    hitter.capture_return_anchor(position, 1.0)
  assert hitter.return_anchor is None


def test_direct_policy_entry_captures_first_observation(hitter):
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.0)
  state = state_for(hitter, y=-0.82)
  hitter.step(state, None, 100.02)
  np.testing.assert_array_equal(hitter.return_anchor, state.root_pos)
  np.testing.assert_array_equal(hitter.debug["goal_base_xy"], state.root_pos[:2])

"""Offline G1 planner transport, contract and controller regression checks.

Run: .venv/bin/python -m pytest deploy/validation/hitter_planner_checks.py
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from deploy.hope_g1_hitter.configuration import load_deployment
from deploy.hope_g1_hitter.controller import HitterController
from deploy.io.g1_planner import command_packet
from deploy.racket_command import decode_g1_planner_command


@pytest.fixture(scope="module")
def deployment():
  torch.set_num_threads(1)
  directory = Path(__file__).resolve().parents[1] / "hope_g1_hitter"
  options = yaml.safe_load((directory / "configs/rally.yaml").read_text())
  runtime = yaml.safe_load((directory / "configs/real.yaml").read_text())
  model = load_deployment(
    directory / "artifacts/hitter_extended_20261103_10700",
    ball_config=options["ball"],
    world_config=options["world"],
    station=options["hitter"]["station"],
  )
  return model, runtime, options


@pytest.fixture
def hitter(deployment):
  return HitterController(*deployment)


def packet_for(hitter, side=0):
  velocity = hitter.geometry.velocity_boxes[side].mean(axis=1)
  return dict(
    schema="g1_1v1_v1",
    runtime_mode=hitter.runtime_mode,
    physics_sha256=hitter.physics_profile.source_sha256,
    contract_sha256=hitter.contract_hash,
    session_id=hitter.session_id,
    command_id=1,
    state_sequence=1,
    valid=True,
    side=side,
    station=hitter.station.tolist(),
    position=hitter.geometry.centre(side, hitter.station).tolist(),
    velocity=velocity.tolist(),
    normal=(velocity / np.linalg.norm(velocity)).tolist(),
    time_to_strike=0.6,
    bridge_monotonic=100.0,
    stamp=1000.1,
    source_ball_time=1000.0,
    predicted_bounces=1,
  )


def robot_state(hitter):
  return SimpleNamespace(
    q=hitter.controller.default.copy(),
    dq=np.zeros(29),
    gyro=np.zeros(3),
    root_pos=np.r_[hitter.station, 0.0],
    root_quat=np.array([1.0, 0, 0, 0]),
  )


@pytest.mark.parametrize("side", [0, 1])
def test_actor_accepts_both_sides_and_holds_follow(hitter, side):
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.0)
  packet = packet_for(hitter, side)
  # Accept an off-centre station so this also checks post-swing behaviour.
  packet["station"][1] += 0.1
  packet["position"] = hitter.geometry.centre(side, packet["station"]).tolist()
  hitter.update_external_planner_command(packet, 100.02)
  state = robot_state(hitter)
  target = hitter.step(state, None, 100.03)
  assert target.shape == (29,) and np.isfinite(target).all()
  assert hitter.controller.debug["actor_obs"].shape == (104,)
  assert hitter.lifecycle.state == "EXECUTE" and hitter.shot.side == side
  np.testing.assert_array_equal(hitter.shot.velocity, packet["velocity"])
  for now in (100.2, 100.5, 100.7):
    hitter.update_external_planner_command(packet, now)
    hitter.step(state, None, now)
    assert hitter.shot.strike_at == pytest.approx(100.6)
  assert hitter.controller.debug["time_to_strike"] == pytest.approx(-0.1)
  # A failed new solve must not abort the accepted shot's follow-through.
  hitter.update_external_planner_command(
    dict(packet, valid=False, command_id=2), 100.71
  )
  hitter.step(state, None, 100.72)
  assert hitter.shot.strike_at == pytest.approx(100.6)
  np.testing.assert_array_equal(hitter.station, packet["station"])
  assert hitter.geometry.follow[side] == pytest.approx([1.3, 1.2][side])
  # The old 0.5 s cutoff must no longer end the swing.
  hitter.step(state, None, 101.2)
  assert hitter.shot is not None
  assert hitter.controller.debug["time_to_strike"] == pytest.approx(-0.6)
  hitter.step(state, None, 100.6 + hitter.geometry.follow[side] - 0.01)
  assert hitter.shot is not None
  last_action = hitter.controller.previous_joint.copy()
  hitter.step(state, None, 100.6 + hitter.geometry.follow[side] + 0.02)
  np.testing.assert_array_equal(
    hitter.controller.debug["actor_obs"][61:90], last_action
  )
  assert hitter.shot is None and hitter.lifecycle.state == "ARMED"
  assert hitter.completed == 1 and hitter.recovered == 0
  for delay in (0.1, 0.5, 2.0):
    hitter.step(state, None, 100.6 + hitter.geometry.follow[side] + delay)
    np.testing.assert_array_equal(hitter.station, packet["station"])
    assert hitter.completed == 1 and hitter.recovered == 0
    np.testing.assert_array_equal(hitter.planner_state()["station"], packet["station"])
    np.testing.assert_allclose(
      hitter.controller.debug["goal_position"],
      hitter.geometry.centre(side, hitter.station),
    )
    assert hitter.controller.debug["time_to_strike"] == hitter.geometry.windup[side]
    np.testing.assert_array_equal(
      hitter.controller.debug["goal_base_xy"], hitter.station
    )


@pytest.mark.parametrize(
  "override,reason",
  [
    ({"schema": "legacy"}, "wrong_planner_schema"),
    ({"contract_sha256": "old"}, "wrong_policy_contract"),
    ({"runtime_mode": "sim"}, "wrong_runtime_mode"),
    ({"physics_sha256": "old"}, "wrong_physics_profile"),
    ({"valid": False}, "ros_planner_published_invalid_command"),
    ({"position": None}, "malformed_ros_planner_command"),
    ({"velocity": [1, 2]}, "malformed_ros_planner_command"),
    ({"normal": [0, 0, 0]}, "malformed_ros_planner_command"),
    ({"time_to_strike": float("nan")}, "malformed_ros_planner_command"),
    ({"time_to_strike": -0.1}, "malformed_ros_planner_command"),
    ({"station": [float("nan"), -0.45]}, "invalid_g1_station_or_side"),
    ({"side": 2}, "invalid_g1_station_or_side"),
    ({"command_id": "bad"}, "malformed_g1_command"),
  ],
)
def test_reject_bad_new_command(hitter, override, reason):
  packet = packet_for(hitter)
  hitter.update_external_planner_command(packet, 100.01)
  hitter.update_external_planner_command(dict(packet, command_id=2) | override, 100.02)
  assert hitter.external_planner_command is None
  assert hitter.external_planner_error == reason


def test_simulation_completion_does_not_require_hitter_recovery(hitter):
  from deploy.g1_pingpong.simulation_runtime import finished_episodes

  hitter.completed, hitter.recovered = 2, 0
  assert finished_episodes(hitter) == 2
  legacy = SimpleNamespace(completed=2, recovered=1)
  assert finished_episodes(legacy) == 1


def test_previous_session_is_ignored_and_restart_uses_configured_station(hitter):
  old = packet_for(hitter)
  hitter.station = np.array([-0.65, -1.1])
  hitter.reset_manual_policy(100.0)
  hitter.lifecycle.state = "STABLE"
  hitter.start_policy(hitter.controller.default, 100.2)
  hitter.update_external_planner_command(old, 100.21)
  assert hitter.external_planner_command is None
  state = robot_state(hitter)
  state.root_pos += [0.1, 0.2, 0.05]
  hitter.step(state, None, 101.0)
  np.testing.assert_array_equal(hitter.station, hitter.geometry.home_station)
  np.testing.assert_array_equal(hitter.planner_state()["station"], hitter.station)
  assert hitter.phase == "IDLE"
  assert not hasattr(hitter, "capture_return_anchor")


@pytest.mark.parametrize("mode", ["sim", "real"])
@pytest.mark.parametrize("during_follow", [False, True])
def test_next_ball_can_replace_target_without_waiting_for_arrival(
  deployment, mode, during_follow
):
  model, runtime, options = deployment
  options = dict(
    options,
    runtime_mode=mode,
    planner_transport="embedded" if mode == "sim" else "ros",
  )
  hitter = HitterController(model, runtime, options)
  hitter.lifecycle.state = "ARMED"
  state = robot_state(hitter)
  packet = packet_for(hitter)
  packet["station"][1] = -0.5
  packet["position"] = hitter.geometry.centre(0, packet["station"]).tolist()
  hitter.update_external_planner_command(packet, 100.0)
  hitter.step(state, None, 100.02)
  np.testing.assert_array_equal(hitter.planner_state()["station"], packet["station"])
  now = 101.2 if during_follow else 102.0
  hitter.step(state, None, now)
  assert hitter.phase == ("SWING" if during_follow else "IDLE")
  np.testing.assert_array_equal(hitter.station, packet["station"])
  # Body still far away: a new target is reachable only from the held target Y.
  state.root_pos[1] = -1.1
  packet = packet_for(hitter, side=1)
  packet.update(command_id=2, bridge_monotonic=now, station=[-0.65, -0.4])
  packet["position"] = hitter.geometry.centre(1, packet["station"]).tolist()
  hitter.update_external_planner_command(packet, now)
  hitter.step(state, None, now + 0.02)
  assert hitter.lifecycle.state == "EXECUTE" and hitter.side == 1
  assert hitter.recovered == 0
  np.testing.assert_array_equal(hitter.station, packet["station"])
  np.testing.assert_array_equal(hitter.planner_state()["station"], packet["station"])
  # Actual position is still used by the policy, only the reachability reference changes.
  np.testing.assert_allclose(
    hitter.controller.debug["actor_obs"][95:97], hitter.station - state.root_pos[:2]
  )


def test_reject_command_beyond_current_target_station_step(hitter):
  hitter.lifecycle.state = "ARMED"
  state = robot_state(hitter)
  packet = packet_for(hitter)
  packet["station"][1] = -1.2
  packet["position"] = hitter.geometry.centre(0, packet["station"]).tolist()
  state.root_pos[1] = -1.1
  hitter.update_external_planner_command(packet, 100.02)
  hitter.step(state, None, 100.03)
  assert hitter.shot is None
  assert hitter.debug["decision"]["attempts"][0]["reason"] == "station_outside_workspace"
  # The measured pelvis is close to the request; the target station is not.
  assert hitter.station[1] == pytest.approx(-.7625)


@pytest.mark.parametrize("target_y", [-1.8, 0.2])
@pytest.mark.parametrize("side", [0, 1])
def test_reject_lateral_target_outside_training_box(hitter, target_y, side):
  packet = packet_for(hitter, side)
  packet["position"][1] = target_y
  hitter.update_external_planner_command(packet, 100.02)
  shot, diagnostics = hitter.select(100.03)
  assert shot is None and diagnostics[0]["reason"] == "position_outside_training_box"


@pytest.mark.parametrize("station_y", [-1.2501, -.3499])
def test_reject_station_outside_global_bounds(hitter, station_y):
  hitter.station[1] = -1.1 if station_y < -1 else -.5
  packet = packet_for(hitter)
  packet["station"][1] = station_y
  packet["position"] = hitter.geometry.centre(0, packet["station"]).tolist()
  hitter.update_external_planner_command(packet, 100.02)
  shot, diagnostics = hitter.select(100.03)
  assert shot is None and diagnostics[0]["reason"] == "station_outside_workspace"


def test_embedded_planning_and_ros_feedback_use_current_target(deployment):
  model, runtime, options = deployment
  hitter = HitterController(
    model, runtime, dict(options, planner_transport="embedded", runtime_mode="sim")
  )
  calls = []
  hitter.embedded_observer = SimpleNamespace(
    update=lambda now, station, side: calls.append(station.copy())
  )
  hitter.lifecycle.state = "ARMED"
  hitter.station = np.array([-0.65, -0.9])
  state = robot_state(hitter)
  state.root_pos[:2] = [-0.7, -0.6]
  hitter.step(state, None, 100.0)
  np.testing.assert_allclose(calls[0], [-0.65, -0.9])
  np.testing.assert_array_equal(calls[0], hitter.planner_state()["station"])


@pytest.mark.parametrize(
  "field,value,reason",
  [
    ("normal", [1, 0, 0], "normal_velocity_mismatch"),
    ("velocity", [10, 0, 0], "velocity_outside_training_box"),
    ("position", [0, 0, 1], "height_outside_training_box"),
    ("station", [0, -0.7625], "position_outside_training_box"),
  ],
)
def test_geometry_rejects_without_clipping(hitter, field, value, reason):
  hitter.update_external_planner_command(packet_for(hitter) | {field: value}, 100.02)
  shot, diagnostics = hitter.select(100.03)
  assert shot is None and diagnostics[0]["reason"] == reason


def test_station_is_received_without_resolving(hitter):
  packet = packet_for(hitter)
  packet["station"][1] -= 0.1
  packet["position"][1] -= 0.1
  hitter.update_external_planner_command(packet, 100.02)
  shot, _ = hitter.select(100.03)
  np.testing.assert_array_equal(shot.station, packet["station"])
  assert hitter.select(100.04)[0] is None  # Consume once.


def test_expired_and_stale_commands(hitter):
  packet = packet_for(hitter)
  hitter.update_external_planner_command(packet, 100.59)
  assert hitter.select(100.61)[1][0]["reason"] == "time_window"
  hitter.update_external_planner_command(dict(packet, command_id=2), 100.01)
  assert hitter.select(100.3)[1][0]["reason"] == "planner_transport_stale"


def test_ros_and_udp_latency_do_not_extend_deadline(hitter):
  def xyz(v):
    return SimpleNamespace(x=v[0], y=v[1], z=v[2])

  packet = packet_for(hitter)
  msg = SimpleNamespace(
    header=SimpleNamespace(
      frame_id="world", stamp=SimpleNamespace(sec=1000, nanosec=200000000)
    ),
    runtime_mode=hitter.runtime_mode,
    physics_sha256=hitter.physics_profile.source_sha256,
    contract_sha256=hitter.contract_hash,
    session_id=hitter.session_id,
    command_id=7,
    state_sequence=8,
    valid=True,
    reason="command_updated",
    side=0,
    strike_time=1000.6,
    source_ball_time=1000.0,
    source_ball_position=xyz([1, -0.8, 0.2]),
    position=xyz(packet["position"]),
    velocity=xyz(packet["velocity"]),
    normal=xyz(packet["normal"]),
    station=xyz([*packet["station"], 0]),
    ball_velocity_outgoing=xyz([4, 0, 2]),
    ball_velocity_requested=xyz([4, 0, 2]),
    outgoing_velocity_residual=0.0,
    total_bounces=1,
  )
  converted = command_packet(
    msg,
    dict(planner_command_source_frame="world", world_frame="hope_table"),
    ros_now=1000.25,
    monotonic_now=100.25,
  )
  command, error = decode_g1_planner_command(
    converted, 100.3, contract_hash=hitter.contract_hash, session_id=hitter.session_id
  )
  assert error == "" and command.deadline == pytest.approx(100.6)
  assert command.deadline - 100.3 == pytest.approx(0.3)


def test_full_mocap_orientation_and_new_racket_geometry(deployment):
  from deploy.backends.unitree import root_orientation

  model, runtime, options = deployment
  assert runtime["root_orientation_source"] == "mocap"
  q = np.array([0.8, 0.2, 0.4, 0.4])
  q /= np.linalg.norm(q)
  np.testing.assert_array_equal(root_orientation(q, np.array([1, 0, 0, 0]), runtime), q)
  np.testing.assert_array_equal(options["ball"]["racket_offset"], [0.242, 0.003, 0])
  assert model.settings["iteration"] == 10700


def test_suspended_fixture_preserves_configured_station(deployment):
  from deploy.backends.mujoco import MujocoBackend
  from deploy.g1_pingpong.simulation_runtime import MujocoRuntime

  model, _, options = deployment
  robot = MujocoBackend(model.model, model.settings)
  runtime = MujocoRuntime.__new__(MujocoRuntime)
  runtime.robot = robot
  runtime.support_active = True
  runtime.landing_started = None
  runtime.support_z = 0.9
  runtime.support_xy = robot.data.qpos[robot.free_qpos : robot.free_qpos + 2].copy()
  runtime.support()
  actual = (
    robot.state().root_pos[:2]
    + np.asarray(model.settings["simulation_world_translation"])[:2]
  )
  np.testing.assert_allclose(actual, options["hitter"]["station"], atol=1e-12)


def test_new_simulated_ball_clears_bounce_count_without_resetting_actor(hitter):
  from hope_planner_g1.observation import BallObserver
  from hope_planner_g1.planner import G1Planner

  hitter.embedded_observer = BallObserver(G1Planner(hitter.cfg["hitter"]))
  hitter.embedded_observer.bounces = 2
  hitter.embedded_observer.observe(1.0, [1.0, -0.8, 0.1])
  action = np.arange(29, dtype=np.float32)
  hitter.controller.previous_joint = action.copy()
  hitter.side = 1
  shot = object()
  hitter.shot = shot
  hitter.reset_ball_observations()
  assert hitter.embedded_observer.bounces == 0
  assert hitter.embedded_observer.last_sample is None
  assert hitter.shot is shot and hitter.side == 1
  np.testing.assert_array_equal(hitter.controller.previous_joint, action)

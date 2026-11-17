"""Hardware-free checks for the two independent policy UDP boundaries."""

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from deploy.io.shared_ball import SharedBallCache
from deploy.io.transport import UDPMocap

CONFIG_DIR = Path(__file__).resolve().parents[1] / "hope_g1_hitter/configs"


def config(player="P1"):
  cfg = yaml.safe_load((CONFIG_DIR / f"mocap_dual_{player.lower()}.yaml").read_text())
  cfg["udp"] = {"host": "127.0.0.1", "port": 0}
  return cfg


def estimate(player="P1", **changes):
  packet = dict(
    kind="ball_estimate",
    schema="g1_duel_ball_v1",
    frame="hope_table",
    robot_id=player,
    valid=True,
    position=[1.4, -0.8, 0.3],
    velocity=[-3.0, 0.1, -1.0],
    stamp=100.0,
    emitted_monotonic=10.0,
    trajectory_epoch=7,
    receiving_side="P1",
  )
  return {**packet, **changes}


def root(player="P1", **changes):
  return dict(
    kind="root",
    frame="hope_table",
    robot_id=player,
    stamp=100.0,
    position=[-0.65, -0.76, 0.8],
    quaternion_wxyz=[1, 0, 0, 0],
    **changes,
  )


def command(player="P1", **changes):
  return dict(
    kind="racket_command",
    frame="hope_table",
    robot_id=player,
    stamp=100.0,
    position=[0, 0, 0],
    velocity=[1, 0, 0],
    normal=[1, 0, 0],
    time_to_strike=0.5,
    **changes,
  )


def accept(receiver, packet, *, wall=100.01, mono=10.01):
  receiver.accept(packet, wall_time=wall, monotonic_time=mono)


def test_shared_ball_uses_host_velocity_and_source_time_without_a_fit(monkeypatch):
  def unexpected_estimator(**kwargs):
    raise AssertionError("A dual receiver must not instantiate a ball estimator")

  monkeypatch.setattr("deploy.io.transport.BallEstimator", unexpected_estimator)
  receiver = UDPMocap(config())
  try:
    accept(receiver, estimate())
    ball = receiver.ball.get(10.01)
    np.testing.assert_array_equal(ball.velocity, [-3.0, 0.1, -1.0])
    assert ball.stamp == 100.0
    assert ball.received == pytest.approx(10.0)
    assert receiver.ball.drain_observations() == []
    assert receiver.ball.get(10.101) is None
    assert receiver.ball.diagnostics() == {
      "trajectory_epoch": 7,
      "receiving_side": "P1",
    }
    ball.position[:] = 99
    np.testing.assert_array_equal(receiver.ball.get(10.01).position, [1.4, -0.8, 0.3])
  finally:
    receiver.close()


@pytest.mark.parametrize("packet", [estimate("P2"), root("P2"), command("P2")])
def test_robot_identity_checked_on_every_packet(packet):
  receiver = UDPMocap(config())
  try:
    with pytest.raises(ValueError, match="robot_id"):
      accept(receiver, packet)
    del packet["robot_id"]
    with pytest.raises(ValueError, match="robot_id"):
      accept(receiver, packet)
    assert receiver.ball.get(10.01) is None
    assert receiver.root.get(10.01) is None
    assert receiver.latest_planner_command() == (None, None)
  finally:
    receiver.close()


def test_invalid_estimate_clears_only_this_player_ball_cache():
  p1, p2 = UDPMocap(config("P1")), UDPMocap(config("P2"))
  try:
    for player, receiver in (("P1", p1), ("P2", p2)):
      accept(receiver, estimate(player))
      accept(receiver, root(player))
      accept(receiver, command(player))
    accept(
      p1,
      estimate(
        valid=False,
        position=None,
        velocity=None,
        stamp=None,
        emitted_monotonic=10.02,
        receiving_side=None,
      ),
      mono=10.03,
    )
    assert p1.ball.get(10.03) is None
    assert p2.ball.get(10.03) is not None
    assert p1.root.get(10.03) is not None
    assert p1.latest_planner_command()[0] == command()
  finally:
    p1.close()
    p2.close()


def test_reordered_duplicate_and_stale_estimates_cannot_refresh_cache():
  cache = SharedBallCache(config())
  cache.accept(estimate(), wall_time=100.01, monotonic_time=10.01)
  assert not cache.accept(
    estimate(emitted_monotonic=10.04), wall_time=100.05, monotonic_time=10.05
  )
  assert cache.get(10.101) is None  # A newer UDP arrival did not renew the sample.
  assert not cache.accept(
    estimate(
      valid=False, stamp=None, position=None, velocity=None, emitted_monotonic=10.03
    ),
    wall_time=100.05,
    monotonic_time=10.05,
  )
  assert cache.get(10.05) is not None
  with pytest.raises(ValueError, match="source time"):
    cache.accept(
      estimate(stamp=99.0, emitted_monotonic=10.06),
      wall_time=100.07,
      monotonic_time=10.07,
    )
  with pytest.raises(ValueError, match="emission time"):
    cache.accept(
      estimate(stamp=100.07, emitted_monotonic=9.0),
      wall_time=100.07,
      monotonic_time=10.07,
    )
  cache.accept(
    estimate(
      valid=False, stamp=None, position=None, velocity=None, emitted_monotonic=10.07
    ),
    wall_time=100.08,
    monotonic_time=10.08,
  )
  assert not cache.accept(
    estimate(stamp=100.06, emitted_monotonic=10.06),
    wall_time=100.08,
    monotonic_time=10.08,
  )
  assert cache.get(10.08) is None
  # A planner restart may reset its diagnostic epoch; monotonic emission orders it.
  assert cache.accept(
    estimate(stamp=100.09, emitted_monotonic=10.09, trajectory_epoch=0),
    wall_time=100.1,
    monotonic_time=10.1,
  )


def test_shared_input_rejects_raw_ball_and_wrong_schema():
  receiver = UDPMocap(config())
  try:
    with pytest.raises(ValueError, match="Raw ball"):
      accept(receiver, estimate(kind="ball"))
    with pytest.raises(ValueError, match="schema"):
      accept(receiver, estimate(schema="unrelated"))
    with pytest.raises(KeyError, match="emitted_monotonic"):
      packet = estimate()
      del packet["emitted_monotonic"]
      accept(receiver, packet)
  finally:
    receiver.close()


def test_udp_socket_demultiplexes_and_feedback_includes_own_identity():
  feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  feedback.bind(("127.0.0.1", 0))
  feedback.settimeout(1.0)
  cfg = config("P2")
  cfg["planner_feedback"] = dict(
    zip(("host", "port"), feedback.getsockname(), strict=True)
  )
  receiver = UDPMocap(cfg)
  sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    now, mono = time.time(), time.monotonic()
    sender.sendto(
      json.dumps(estimate("P1", stamp=now, emitted_monotonic=mono)).encode(),
      receiver.socket.getsockname(),
    )
    sender.sendto(
      json.dumps(estimate("P2", stamp=now, emitted_monotonic=mono)).encode(),
      receiver.socket.getsockname(),
    )
    deadline = time.monotonic() + 1.0
    while receiver.accepted < 1 and time.monotonic() < deadline:
      time.sleep(0.002)
    assert receiver.accepted == 1
    assert receiver.rejected == 1
    receiver.publish_planner_state(
      {"robot_id": "P1", "session_id": "p2-session", "enabled": True}
    )
    packet = json.loads(feedback.recv(4096))
    assert packet["robot_id"] == "P2"
    assert packet["session_id"] == "p2-session"
    assert packet["generated_monotonic"] <= time.monotonic()
  finally:
    receiver.close()
    sender.close()
    feedback.close()


def test_legacy_raw_input_still_fits_and_needs_no_robot_identity():
  cfg = yaml.safe_load((CONFIG_DIR / "mocap.yaml").read_text())
  cfg["udp"] = {"host": "127.0.0.1", "port": 0}
  receiver = UDPMocap(cfg)
  try:
    for index in range(6):
      t = index * 0.01
      packet = dict(
        kind="ball",
        frame="hope_table",
        stamp=100 + t,
        position=[1.4 - 3 * t, -0.8, 1.0],
      )
      accept(receiver, packet, wall=100 + t, mono=10 + t)
    np.testing.assert_allclose(
      receiver.ball.get(10.05).velocity, [-3, 0, 0], atol=1e-10
    )
    assert len(receiver.ball.drain_observations()) == 6
  finally:
    receiver.close()


def test_real_dual_entry_refuses_the_host_namespace_before_sdk(monkeypatch, capsys):
  from deploy.hope_g1_hitter import deploy_agent

  exists = Path.exists

  def namespace_absent(path):
    return False if str(path) == "/run/netns/g1_p1" else exists(path)

  def unexpected_sdk():
    raise AssertionError("SDK must not initialize before the namespace guard")

  monkeypatch.setattr(Path, "exists", namespace_absent)
  monkeypatch.setattr("deploy.backends.sdk.require_sdk", unexpected_sdk)
  monkeypatch.setattr(deploy_agent.signal, "signal", lambda *_args: None)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "deploy_agent",
      "--env",
      "real",
      "--no-chain-record",
      "--mocap-config",
      str(CONFIG_DIR / "mocap_dual_p1.yaml"),
    ],
  )
  with pytest.raises(SystemExit) as error:
    deploy_agent.main()
  assert error.value.code == 2
  assert "inside g1_p1" in capsys.readouterr().err

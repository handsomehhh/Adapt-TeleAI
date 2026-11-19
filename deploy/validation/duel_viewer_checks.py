"""Offline dual-scene transforms, stale-data isolation and recording handoff."""

import hashlib
import json
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from deploy.io.live_visualization import LiveVisualization
from deploy.io.viewer_snapshot import boot_id, write_snapshot
from deploy.tools.duel_viewer import (
  DuelScene, ball_position, canonical_position, robot_state,
)
from deploy.validation.live_visualization_checks import deployment, profile, robot_row


def packet(side, now):
  return dict(schema="dual_robot_view_v1", robot_id=side, boot_id=boot_id(),
              time=now, data=robot_row(now))


def recording(run, side, now):
  directory = run / side / "policy/visualization"
  directory.mkdir(parents=True)
  dep = deployment()
  model = directory / "scene.mjb"
  mujoco.mj_saveModel(dep.model, str(model))
  (directory / "metadata.json").write_text(json.dumps(dict(
    robot_id=side, boot_id=boot_id(),
    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    joint_names=dep.settings["robot"]["joint_names"],
    world_translation=dep.settings["simulation_world_translation"],
  )))
  write_snapshot(directory / "latest_state.json", packet(side, now))
  return directory


def test_p2_inverse_transform_is_involution_and_preserves_height():
  p = [-.65, -.7625, .2]
  np.testing.assert_allclose(canonical_position(p, "P1"), p)
  np.testing.assert_allclose(canonical_position(p, "P2"), [3.39, -.7625, .2])
  np.testing.assert_allclose(canonical_position(canonical_position(p, "P2"), "P2"), p)


@pytest.mark.parametrize("change", [
  lambda p: p.update(robot_id="P2"),
  lambda p: p.update(boot_id="previous-boot"),
  lambda p: p.update(time=9.),
  lambda p: p.update(time=11.),
  lambda p: p["data"].update(dds_state_age=.3),
  lambda p: p["data"].update(root_age=.3),
  lambda p: p["data"].update(q=[0.] * 28),
  lambda p: p["data"].update(root_quat=[0.] * 4),
  lambda p: p["data"].update(root_age=float('nan')),
])
def test_invalid_or_wrong_side_robot_is_hidden(change):
  p = packet("P1", 10.)
  assert robot_state(p, "P1", 10.05, boot_id()).robot_valid(10.05)
  change(p)
  assert not robot_state(p, "P1", 10.05, boot_id()).robot_valid(10.05)


def test_snapshot_replace_is_atomic_and_cannot_refresh_old_ball(tmp_path):
  path = tmp_path / "ball.json"
  p = dict(schema="dual_ball_view_v1", boot_id=boot_id(), frame="canonical_P1",
           time=10., position=[1., -.4, .2])
  write_snapshot(path, p)
  assert ball_position(json.loads(path.read_text()), 10.1, boot_id()) is not None
  write_snapshot(path, p)  # File modification time is irrelevant to freshness.
  assert ball_position(json.loads(path.read_text()), 10.3, boot_id()) is None
  with pytest.raises(ValueError):
    write_snapshot(path, dict(p, time=float('nan')))
  assert json.loads(path.read_text()) == p
  assert list(tmp_path.glob('*.tmp')) == []


def test_two_robots_one_ball_and_table_and_independent_staleness(tmp_path):
  from viser.transforms import SO3

  now = time.monotonic()
  recording(tmp_path, "P1", now)
  ball = tmp_path / "ball.json"
  write_snapshot(ball, dict(schema="dual_ball_view_v1", boot_id=boot_id(),
                            frame="canonical_P1", time=now, position=[1., -.4, .3]))
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
  scene = DuelScene(tmp_path, ball, port=port)
  try:
    scene.update(now)
    assert set(scene.robots) == {"P1"} and scene.ball.visible
    assert scene.robots['P1'].label.visible
    recording(tmp_path, "P2", now)
    scene.retry_at['P2'] = 0
    scene.update(now + .01)
    assert set(scene.robots) == {"P1", "P2"}
    p1, p2 = scene.robots['P1'], scene.robots['P2']
    np.testing.assert_allclose(p1.data.qpos[p1.addresses], p2.data.qpos[p2.addresses])
    for geom, handle in p1.handles.items():
      np.testing.assert_allclose(p2.handles[geom].position,
                                 canonical_position(handle.position, 'P2'), atol=1e-6)
      np.testing.assert_allclose(SO3(p2.handles[geom].wxyz).as_matrix(),
                                 np.diag([-1., -1., 1.]) @ SO3(handle.wxyz).as_matrix(), atol=1e-6)
    assert all(p1.model.geom_bodyid[g] != 0 for g in p1.handles)  # No duplicate table.
    write_snapshot(tmp_path / 'P2/policy/visualization/latest_state.json', packet('P2', now + .3))
    scene.update(now + .31)
    assert not p1.label.visible and p2.label.visible
    assert not scene.ball.visible
  finally:
    scene.close()


def test_worker_exports_measured_state_without_own_web_server(tmp_path):
  sink = LiveVisualization(deployment(), dict(physics_profile_report=profile()),
                           tmp_path / 'P1/policy', host='127.0.0.1', port=8082,
                           headless=True, robot_id='P1')
  try:
    t = time.monotonic()
    sink.publish('state', robot_row(t), received=t)
    path = sink.directory / 'latest_state.json'
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
      time.sleep(.02)
    value = json.loads(path.read_text())
    assert value['robot_id'] == 'P1' and value['time'] == t
    assert value['data']['q'] == robot_row(t)['q']
    assert value['boot_id'] == boot_id()
  finally:
    sink.close()


def test_viewer_follows_independent_actor_restart_without_restart_or_matching_ids(tmp_path):
  runtime = tmp_path / 'runtime'
  now = time.monotonic()
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
  scene = DuelScene(runtime, tmp_path / 'no-ball-yet.json', port=port)
  try:
    scene.update(now)  # Viewer can start even before the runtime directory exists.
    assert not scene.robots
    runtime.mkdir()
    first = recording(tmp_path / 'old', 'P1', now)
    other = recording(tmp_path / 'independent', 'P2', now)
    (runtime / 'P1').symlink_to(first.parent.parent)
    (runtime / 'P2').symlink_to(other.parent.parent)
    scene.update(now + .01)
    old_robot, p2 = scene.robots['P1'], scene.robots['P2']
    assert old_robot.label.visible and p2.label.visible
    # P1 restarts too quickly to observe a gap; replace the fixed link atomically.
    newer = recording(tmp_path / 'new', 'P1', now + .02)
    new_packet = packet('P1', now + .02)
    new_packet['data']['root_pos'][0] += .5
    write_snapshot(newer / 'latest_state.json', new_packet)
    (runtime / 'next-P1').symlink_to(newer.parent.parent)
    (runtime / 'next-P1').replace(runtime / 'P1')
    scene.update(now + .03)
    assert scene.robots['P1'] is not old_robot
    assert scene.robots['P1'].directory == newer
    assert scene.robots['P2'] is p2 and p2.label.visible
    np.testing.assert_allclose(scene.robots['P1'].label.position,
                               np.array(new_packet['data']['root_pos']) + [0, 0, .65])
    (runtime / 'P1').unlink()
    scene.update(now + .04)
    assert 'P1' not in scene.robots and scene.robots['P2'] is p2
    assert first.exists() and newer.exists()  # History is never removed.
  finally:
    scene.close()

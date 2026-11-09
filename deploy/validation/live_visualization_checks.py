"""Read-only telemetry integration: bounded handoff, clocks, rendering and replay."""

import json
import queue
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from deploy.io.live_visualization import LiveVisualization
from deploy.tools.real_viewer import DisplayState, Scene, read_events


def deployment():
  bodies = "".join(
    f'<body pos="0 0 .03"><joint name="robot/j{i}" type="hinge"/>'
    '<geom type="sphere" size=".02" mass=".1"/>'
    for i in range(29)
  )
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom name="table" type="box" size="1 .5 .02"/>'
    '<body name="robot/pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
    + bodies
    + "</body>" * 30
    + "</worldbody></mujoco>"
  )
  return SimpleNamespace(
    model=model,
    checkpoint=Path("fake.pt"),
    settings=dict(
      robot=dict(joint_names=[f"j{i}" for i in range(29)]),
      simulation_world_translation=[-0.65, -0.7625, -0.76],
    ),
  )


def profile():
  from deploy.hope_g1_hitter.planner_source import use_sibling_planner

  use_sibling_planner()
  from hope_planner_g1.profiles import load_profile

  return load_profile("real").report()


def robot_row(t):
  return dict(
    time=t,
    state="EXECUTE",
    q=[0.1] * 29,
    root_pos=[-0.65, -0.7625, 0.1],
    root_quat=[1, 0, 0, 0],
    root_age=0.01,
    dds_state_age=0.005,
  )


def test_full_queue_never_blocks_or_mutates_control_row():
  sink = LiveVisualization.__new__(LiveVisualization)
  sink.events = queue.Queue(maxsize=1)
  sink.accepted = sink.dropped = 0
  row = robot_row(1)
  row["large_unused_diagnostic"] = [0] * 1000
  sink.publish("state", row)
  for _ in range(100):
    sink.publish("state", row)
  assert sink.accepted == 1 and sink.dropped == 100
  assert "large_unused_diagnostic" not in sink.events.get_nowait()["data"]
  assert "large_unused_diagnostic" in row


def test_source_age_and_missing_robot_are_not_displayed_as_fresh():
  state = DisplayState()
  state.apply(dict(kind="state", time=10, data=robot_row(10)))
  assert state.robot_valid(10.1)
  assert not state.robot_valid(10.3)
  state.apply(dict(kind="state", time=10.4, data=dict(state="HALT", q=[0] * 29)))
  assert not state.robot_valid(10.4)
  state.apply(
    dict(
      kind="mocap",
      time=11,
      data=dict(
        kind="ball",
        frame="hope_table",
        stamp=100,
        sample_monotonic=10,
        position=[1, 0, 1],
      ),
    )
  )
  assert state.ball_time == 10
  state.apply(
    dict(
      kind="mocap",
      time=12,
      data=dict(
        kind="ball",
        frame="wrong",
        stamp=101,
        position=[2, 0, 1],
      ),
    )
  )
  assert state.ball == [1, 0, 1]


def test_udp_tap_uses_the_control_sample_clock_and_rejects_wrong_frame():
  from deploy.io.transport import UDPMocap

  received = []
  source = UDPMocap.__new__(UDPMocap)
  source.cfg = dict(
    world_frame="hope_table", source_clock_offset=0, source_max_age=0.15
  )
  source.recorder = None
  source.accepted = 0
  source.ball = SimpleNamespace(update=lambda *args, **kwargs: None)
  source.observer = SimpleNamespace(
    publish=lambda *args, **kwargs: received.append((args, kwargs))
  )
  packet = dict(kind="ball", frame="hope_table", stamp=99.98, position=[1, -0.7, 0.5])
  source.accept(packet, wall_time=100, monotonic_time=10)
  assert received[0][0][1]["sample_monotonic"] == pytest.approx(9.98)
  assert received[0][1]["received"] == 10
  with pytest.raises(ValueError):
    source.accept(dict(packet, frame="map"), wall_time=100, monotonic_time=10)
  with pytest.raises(ValueError):
    source.accept(dict(packet, stamp=90), wall_time=100, monotonic_time=10)
  assert len(received) == 1


def test_process_records_predictions_and_measured_robot_for_exact_replay(tmp_path):
  sink = LiveVisualization(
    deployment(),
    {"physics_profile_report": profile()},
    tmp_path,
    host="127.0.0.1",
    port=8081,
    headless=True,
  )
  try:
    start = time.monotonic()
    sink.publish("state", robot_row(start))
    for i in range(35):
      t = i * 0.01
      sink.publish(
        "mocap",
        dict(
          kind="ball",
          frame="hope_table",
          stamp=100 + t,
          position=[2 - t * 3, -0.7, 1 + t - 4.905 * t * t],
        ),
      )
    sink.publish(
      "mocap",
      dict(
        kind="racket_command",
        frame="hope_table",
        valid=True,
        position=[-0.1, -0.7, 0.2],
        time_to_strike=0.3,
      ),
    )
  finally:
    sink.close()
  directory = sink.directory
  assert not (directory / "worker_error.txt").exists()
  events = list(read_events(directory / "events.jsonl"))
  predictions = [r for r in events if r["kind"] == "prediction"]
  assert len(predictions) >= 2
  state = DisplayState()
  for row in events:
    state.apply(row)
  assert state.robot["q"] == [0.1] * 29
  assert state.command["position"] == [-0.1, -0.7, 0.2]
  assert state.prediction["points"] == predictions[-1]["data"]["points"]
  assert state.reference["points"] == predictions[0]["data"]["points"]
  transport = json.loads((directory / "transport_summary.json").read_text())
  assert transport["dropped"] == 0 and transport["worker_exitcode"] == 0
  assert not transport["forced_stop"]

  # Render the saved model without loading a policy or taking a physics step.
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
  scene = Scene(directory, host="127.0.0.1", port=port, replay=True)
  try:
    state.robot_time = state.now
    scene.update(state, state.now)
    assert np.allclose(scene.data.qpos[scene.root : scene.root + 3], [0, 0, 0.86])
    assert np.allclose(scene.data.qpos[scene.addresses], 0.1)
    assert scene.data.time == 0  # No mj_step, no simulation clock.
    assert np.asarray(scene.lines["latest"].colors).shape == (3,)
    assert scene.lines["latest"].visible
    scene.update(state, state.now + 1)
    assert not scene.ball.visible and not scene.command.visible
    assert scene.data.qpos[scene.root + 2] == -100
  finally:
    scene.close()


def test_truncated_final_event_is_ignored(tmp_path):
  path = tmp_path / "events.jsonl"
  path.write_text('{"kind": "state"}\n{"kind":')
  assert list(read_events(path)) == [{"kind": "state"}]

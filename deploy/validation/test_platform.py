"""Actual first-round models: module isolation, runtime parity and exact replay."""

import numpy as np
import pytest
import torch

from deploy import ROOT
from deploy.artifacts import Artifacts
from deploy.comparison import ReferenceRobot, reference_qpos
from deploy.control import MODES, Controller
from deploy.replay import Replay
from deploy.session import Session
from deploy.world import World
from mjlab.tasks.g1_adapt.runtime import HierarchicalController

RUN = ROOT / "logs/g1_adapt_full/2026-11-02_18-30-26_extended27mm_gpu0_env4096"


@pytest.fixture
def artifacts():
  if not (RUN / "bundle/bundle.json").is_file():
    pytest.skip("First-round local trained models required")
  torch.set_num_threads(2)
  return Artifacts(RUN)


@pytest.mark.parametrize("mode", MODES)
def test_every_module_executes_actual_weights_and_records_fixed_contract(
  artifacts, mode, tmp_path
):
  session = Session(artifacts, artifacts.motion, mode)
  for _ in range(12):
    row = session.step()
    assert np.isfinite(row["qpos"]).all()
    assert len(row["targets"]) == 29
    assert len(row["tracker_obs"]) == 132
    assert len(row["planner_obs"]) == 309
  report = session.save(tmp_path)
  assert report["backend"] == "native_cpu_mujoco"
  assert report["physics_dt"] == 0.005
  assert session.world.model.nu == 29
  with np.load(tmp_path / "trace.npz", allow_pickle=False) as data:
    assert data["qpos"].shape == (12, 36)


def test_tracker_and_mvae_can_be_tested_without_other_models(artifacts):
  tracker = Artifacts(RUN)
  tracker.files = {"tracker": tracker.files["tracker"]}
  session = Session(tracker, tracker.motion, "tracker")
  session.step()
  mvae = Artifacts(RUN)
  mvae.files = {"mvae": mvae.files["mvae"]}
  session = Session(mvae, mvae.motion, "mvae")
  assert session.step()["kinematic"] is True


def test_hierarchy_observations_targets_and_feedback_match_exported_runtime(artifacts):
  world = World()
  controller = Controller(artifacts, artifacts.motion, world.default_q, "hierarchy")
  world.reset(controller.reference)
  runtime = HierarchicalController(RUN / "bundle", device="cpu")
  with torch.inference_mode():
    runtime.reset(world.robot_state)
    for t in (0.02, 0.04, 0.06, 0.3):
      position = torch.tensor([[1.5 - t, 0.0, 1.0]]) if t < 0.3 else None
      targets, info = runtime.step(world.robot_state, position, t)
      actual = controller.step(world.robot_state, info["planner_observation"][:, -101:])
      torch.testing.assert_close(actual, targets, atol=2e-5, rtol=2e-5)
      torch.testing.assert_close(
        controller.info["tracker_observation"],
        info["tracker_observation"],
        atol=2e-5,
        rtol=2e-5,
      )
      torch.testing.assert_close(
        controller.current.history, runtime.generator.history, atol=2e-5, rtol=2e-5
      )


def test_recording_replay_is_exact_and_does_not_step_physics(artifacts, tmp_path):
  session = Session(artifacts, artifacts.motion, "tracker")
  for _ in range(10):
    session.step()
  session.save(tmp_path)
  replay = Replay(tmp_path / "trace.npz")
  for index in (0, 9, 3, 8):
    replay.seek(index)
    np.testing.assert_array_equal(
      replay.world.data.qpos, session.records[index]["qpos"]
    )
    assert replay.world.data.time == 0.0


def test_planner_rejects_mismatched_frozen_tracker(artifacts, tmp_path):
  wrong = tmp_path / "wrong.pt"
  wrong.write_bytes(b"not the trained tracker")
  artifacts.files["tracker"] = wrong
  with pytest.raises(ValueError, match="exact frozen tracker"):
    artifacts.require(["rally"])


def test_continuous_feed_preserves_robot_and_generated_state(artifacts):
  session = Session(artifacts, artifacts.motion, "hierarchy", scenario="continuous")
  for _ in range(4):
    session.step()
  qpos = session.world.data.qpos.copy()
  history = session.controller.current.history.clone()
  previous_action = session.controller.joint_action.clone()
  session.active = False
  session.ball_in_flight = False
  session.ball.done[:] = True
  assert session.feed()
  np.testing.assert_array_equal(session.world.data.qpos, qpos)
  torch.testing.assert_close(session.controller.current.history, history)
  torch.testing.assert_close(session.controller.joint_action, previous_action)


def test_tracker_speed_observation_and_recording_limit(artifacts):
  session = Session(artifacts, artifacts.motion, "tracker")
  session.speed = 0.5
  session.max_record_steps = 2
  first, second = session.step(), session.step()
  assert first["tracker_obs"][-1] == 0.0
  assert second["tracker_obs"][-1] == pytest.approx(-0.01)
  session.step()
  assert session.paused and session.record_limit_reached and len(session.records) == 2


def test_viewer_does_not_change_simulation_results(artifacts, monkeypatch, tmp_path):
  # No socket needed: an in-memory display stub exercises all frame overlays.
  from deploy.viewer import Viewer

  triggers = {}

  class Handle:
    value = False
    hotkey = None

    def on_update(self, fn):
      return fn

    def on_click(self, fn):
      return fn

    def on_trigger(self, fn):
      triggers[self.hotkey] = fn
      return fn

    def __enter__(self):
      return self

    def __exit__(self, *_):
      pass

  class Stub:
    def __getattr__(self, _):
      def add(*args, **kwargs):
        handle = Handle()
        handle.value = kwargs.get("initial_value", False)
        handle.hotkey = kwargs.get("hotkey")
        return handle

      return add

  class Server:
    def __init__(self, **_):
      self.gui, self.scene = Stub(), Stub()

    def on_client_connect(self, fn):
      return fn

  class Scene:
    def __init__(self, *_args, **_kwargs):
      pass

    def update_from_mjdata(self, data):
      pass

  monkeypatch.setattr("deploy.viewer.viser.ViserServer", Server)
  monkeypatch.setattr("deploy.viewer.ViserMujocoScene", Scene)
  left = Session(Artifacts(RUN), artifacts.motion, "hierarchy", seed=123)
  left_rows = [left.step() for _ in range(8)]
  right = Session(Artifacts(RUN), artifacts.motion, "hierarchy", seed=123)
  viewer = Viewer(right)
  for expected in left_rows:
    actual = right.step()
    for fraction in (0.0, 0.25, 0.5, 1.0):
      viewer.update(fraction)
    np.testing.assert_array_equal(actual["qpos"], expected["qpos"])
  # Exercise the same main-thread commands used by the browser buttons.
  expected_motion = right.adjacent_motion(1)
  viewer.commands.put(("mode", "tracker"))
  viewer.commands.put(("adjacent_motion", 1))
  viewer.commands.put(("motion_seek", 0.1))
  viewer.drain(tmp_path)
  viewer.update()
  assert right.paused and right.controller.reference_time == 0.1
  assert right.motion == expected_motion
  assert viewer.motion_picker is not None
  assert not viewer.motion_picker.disabled
  np.testing.assert_array_equal(
    viewer.reference_robot.data.qpos, right.info["reference_qpos"]
  )
  viewer.commands.put(("reset", None))
  viewer.drain(tmp_path)
  viewer.update()
  assert right.controller.reference_time == 0
  np.testing.assert_array_equal(viewer.reference_robot.data.qpos, right.world.data.qpos)
  # Hotkeys only queue work; the simulation thread applies it and syncs the GUI.
  triggers["space"](None)
  assert right.paused
  viewer.drain(tmp_path)
  assert not right.paused and not viewer.pause_handle.value
  right.step()
  triggers["R"](None)
  assert right.controller.reference_time > 0
  viewer.drain(tmp_path)
  assert right.paused and viewer.pause_handle.value
  assert right.controller.reference_time == 0


def test_tracker_switch_and_seek_reset_execution_and_record_clip_identity(
  artifacts, tmp_path
):
  session = Session(artifacts, artifacts.motion, "tracker")
  session.step()
  tracker = session.controller.tracker
  next_motion = session.adjacent_motion(1)
  assert next_motion != session.motion
  session.set_motion(next_motion)
  assert session.paused and session.controller.tracker is tracker
  assert session.motion == next_motion and session.controller.reference_time == 0
  np.testing.assert_allclose(
    session.world.data.qpos, reference_qpos(session.world, session.controller.reference)
  )
  assert session.controller.joint_action.count_nonzero() == 0
  session.seek_motion(session.controller.reference.duration / 2)
  assert session.controller.tracker_delta == 0
  np.testing.assert_allclose(
    session.world.data.qpos, reference_qpos(session.world, session.controller.reference)
  )
  row = session.step()
  assert row["motion"] == str(next_motion) and row["segment"] == 2
  assert row["tracker_obs"][-1] == 0
  report = session.save(tmp_path)
  assert len(report["motion_events"]) == 3
  replay = Replay(tmp_path / "trace.npz")
  replay.seek(1)
  assert replay.info["motion"] == str(next_motion)
  np.testing.assert_array_equal(replay.world.data.qpos, row["qpos"])


def test_tracker_motion_end_pause_and_loop(artifacts):
  session = Session(artifacts, artifacts.motion, "tracker")
  duration = session.controller.reference.duration
  session.seek_motion(duration - 0.01)
  session.paused = False
  session.step()
  assert session.paused and session.tracker_at_end
  assert session.controller.reference_time == duration
  steps = len(session.records)
  session.step()
  assert len(session.records) == steps
  session.tracker_loop, session.paused = True, False
  row = session.step()
  assert not session.paused and not session.tracker_at_end
  assert row["motion_time_s"] == pytest.approx(0.02)
  assert row["tracker_obs"][-1] == 0


def test_invalid_motion_or_seek_preserves_running_tracker(artifacts, tmp_path):
  broken = tmp_path / "broken.npz"
  broken.write_bytes(b"bad clip")
  session = Session(artifacts, artifacts.motion, "tracker")
  session.motion_catalog["bad clip"] = broken
  old_reference = session.controller.reference
  qpos = session.world.data.qpos.copy()
  for target in (-1, float("nan"), session.controller.reference.duration + 1):
    with pytest.raises(ValueError):
      session.seek_motion(target)
  with pytest.raises((ValueError, OSError)):
    session.set_motion(broken)
  assert session.controller.reference is old_reference
  assert session.motion == artifacts.motion
  np.testing.assert_array_equal(session.world.data.qpos, qpos)
  assert any(name.startswith("训练原始动作") for name in session.motion_catalog)
  assert any(name.startswith("物理修正动作") for name in session.motion_catalog)


def test_complete_reference_geometry_includes_racket_and_does_not_modify_plant(
  artifacts,
):
  class Handle:
    pass

  class Scene:
    def add_frame(self, *_args, **_kwargs):
      return Handle()

    def add_label(self, *_args, **_kwargs):
      return Handle()

    def add_mesh_simple(self, _name, vertices, faces, **_kwargs):
      assert len(vertices) > 0 and len(faces) > 0
      return Handle()

  session = Session(artifacts, artifacts.motion, "tracker")
  ghost = ReferenceRobot(Scene(), session.world)
  before = session.world.data.qpos.copy()
  rgba = session.world.model.geom_rgba.copy()
  ids = {ghost.model.body(i).name for i in ghost.handles}
  assert "robot/right_wrist_yaw_link" in ids
  assert "robot/left_ankle_roll_link" in ids
  # The compiler fuses fixed racket links into the wrist body; their geoms remain.
  assert any(
    "racket" in (ghost.model.geom(i).name or "")
    and ghost.model.geom_group[i] == 2
    and ghost.model.geom_bodyid[i] in ghost.handles
    for i in range(ghost.model.ngeom)
  )
  for time in (0.0, session.controller.reference.duration / 2):
    reference = session.controller.reference.at(time)
    qpos = reference_qpos(session.world, reference)
    ghost.update(qpos, np.array([0, 1.4, 0]), opacity=0.35)
    np.testing.assert_array_equal(ghost.data.qpos, qpos)
    for body_id, handle in ghost.handles.items():
      np.testing.assert_array_equal(handle.position, ghost.data.xpos[body_id])
      np.testing.assert_array_equal(handle.wxyz, ghost.data.xquat[body_id])
    np.testing.assert_array_equal(session.world.data.qpos, before)
    np.testing.assert_array_equal(session.world.model.geom_rgba, rgba)
    assert session.world.data.time == 0


def test_continuous_failure_keeps_running_without_reset_or_pause(
  artifacts, monkeypatch
):
  session = Session(artifacts, artifacts.motion, "hierarchy", scenario="continuous")
  session.world.data.qpos[2] = 0.3
  session.world.refresh()
  segment = session.segment

  def forbidden_reset(*_args, **_kwargs):
    pytest.fail("Continuous testing must not reset a failed robot automatically")

  monkeypatch.setattr(session.world, "reset", forbidden_reset)
  monkeypatch.setattr(session.controller, "reset", forbidden_reset)
  for _ in range(300):
    session.step()
    assert not session.paused
  assert session.elapsed == pytest.approx(6)
  assert session.segment == segment
  assert session.records.total == 300
  assert session.fall_events
  assert session.records.ever_fallen
  assert session.trials and session.trials[0]["fallen"]
  assert session.controller.info["policy"] == "standing"


def test_continuous_ball_is_visible_and_moves_after_scoring_finishes(artifacts):
  session = Session(artifacts, artifacts.motion, "ball", scenario="continuous")
  session.ball.done[:] = True
  session.ball.p[:] = torch.tensor([[0.1, 0, 1.0]])
  session.ball.v[:] = torch.tensor([[1.0, 0, -1.0]])
  first = session.ball.p.clone()
  session.step()
  assert not session.active and session.ball_in_flight
  assert session.info["ball_active"]
  assert not torch.equal(first, session.ball.p)
  for _ in range(30):
    session.step()
    if not session.ball_in_flight:
      break
  assert not session.ball_in_flight
  assert session.ball.p[0, 2] == pytest.approx(session.ball.cfg.physics.radius)
  assert session.elapsed > session.trials[0]["duration_s"]


def test_streaming_records_are_bounded_and_replay_keeps_every_frame(
  artifacts, tmp_path
):
  session = Session(artifacts, artifacts.motion, "tracker")
  session.records.directory = tmp_path
  session.records.chunk_steps = 3
  expected = []
  for _ in range(23):
    expected.append(session.step()["qpos"])
    assert len(session.records) < 6
  report = session.save(tmp_path)
  assert report["steps"] == 23 and report["trace_chunks"]
  replay = Replay(tmp_path / "trace.npz")
  assert len(replay.records) == 23
  for index in range(23):
    replay.seek(index)
    np.testing.assert_array_equal(replay.world.data.qpos, expected[index])
  assert not session.paused


def test_ball_observation_matches_training_update_and_resets_per_feed(artifacts):
  session = Session(
    artifacts, artifacts.motion, "hierarchy", noise=0, dropout=0, delay_steps=0
  )
  assert session.ball.cfg.delay_steps == 0
  assert session.estimator is session.ball.estimator
  assert session.estimator.count[0] == 1
  session.step()
  torch.testing.assert_close(session.ball_obs, session.ball.command)
  assert session.estimator.count[0] == 2
  session.active = session.ball_in_flight = False
  assert session.feed()
  assert session.estimator.count[0] == 1
  assert not session.ball.observed_valid[0]


def test_root_angular_velocity_uses_link_axes_even_with_rotated_inertia():
  world = World()
  world.data.qpos[3:7] = [np.cos(0.4), 0, np.sin(0.4), 0]
  world.data.qvel[3:6] = [0.7, -1.1, 0.3]
  world.refresh()
  # A free joint's angular qvel is expressed in body axes, irrespective of inertia.
  torch.testing.assert_close(
    world.robot_state.root_angular_velocity_body,
    torch.tensor([[0.7, -1.1, 0.3]]),
    atol=1e-6,
    rtol=1e-6,
  )


def test_continuous_numeric_fault_is_recorded_without_truncating_physics(
  artifacts, monkeypatch, tmp_path
):
  session = Session(artifacts, artifacts.motion, "hierarchy", scenario="continuous")
  first = session.step()

  def invalid_policy(*_args, **_kwargs):
    raise RuntimeError("Generated reference became non-finite")

  monkeypatch.setattr(session.controller, "step", invalid_policy)
  for _ in range(100):
    row = session.step()
    assert not session.paused
    assert np.isfinite(row["qpos"]).all()
    assert not row["policy_output_valid"]
    np.testing.assert_array_equal(row["targets"], first["targets"])
  assert len(session.control_fault_events) == 1
  assert session.records.total == 101
  report = session.save(tmp_path)
  assert report["control_fault_active"]
  assert report["completed"]

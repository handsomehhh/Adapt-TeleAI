"""Display conversion, control-thread isolation and score/trace invariance."""

import json
import socket
import threading
import urllib.request
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from deploy import viewer
from deploy.benchmarks.planner_snapshot import PlannerSnapshot
from deploy.benchmarks.racket_command.environment import MujocoCommandEnvironment
from deploy.benchmarks.racket_command.sampling import Trial
from deploy.tools.view_real_log import Timeline


def model_data():
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="plane" size="3 3 .1"/>'
    '<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>'
    "</body></worldbody></mujoco>"
  )
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def test_multiframe_seek_uses_recorded_samples_and_clamps():
  times = [0, 0.019, 0.041, 0.061, 0.09, 0.12, 0.16, 0.2, 0.3, 0.4]
  recording = SimpleNamespace(
    duration=times[-1],
    times=times,
    index=lambda t: max(0, np.searchsorted(times, t, side="right") - 1),
  )
  timeline = Timeline(recording)
  for count in (2, 4, 8):
    timeline.position = 0
    timeline.paused = False
    timeline.step_frames(count)
    assert timeline.position == times[count] and timeline.paused
    timeline.step_frames(-count)
    assert timeline.position == 0
  timeline.step_frames(-8)
  assert timeline.position == 0
  timeline.step_frames(80)
  assert timeline.position == times[-1]


def test_tts_countdown_uses_log_time_and_only_available_encounter():
  recording = SimpleNamespace(
    duration=10,
    command=lambda t: dict(received=3.0, deadline=5.0) if 3 <= t < 7 else None,
    encounter=lambda t, command: (
      dict(time=4.97, kind="estimated_hit") if t >= 5.2 else None
    ),
  )
  timeline = Timeline(recording, start=2, paused=True)
  assert timeline.tts_status() is None
  timeline.position = 3
  assert timeline.tts_status()["initial"] == 2
  timeline.position = 4.8
  before = timeline.tts_status()
  assert before["remaining"] == pytest.approx(0.2)
  assert "event_error" not in before
  timeline.advance(100)  # Pausing freezes both the clock and the countdown.
  assert timeline.tts_status() == before
  timeline.position = 5.3
  after = timeline.tts_status()
  assert after["remaining"] == pytest.approx(-0.3)
  assert after["event_error"] == pytest.approx(-0.03)
  timeline.position = 4.8  # Backward seek must not retain future hit evidence.
  assert timeline.tts_status() == before


@pytest.fixture
def port():
  try:
    with socket.socket() as probe:
      probe.bind(("127.0.0.1", 0))
      return probe.getsockname()[1]
  except PermissionError:
    pytest.skip("This integration test needs a localhost listening socket")


def test_debug_geometry_preserves_capsule_endpoints_arrow_and_opacity():
  model, _ = model_data()
  scene = mujoco.MjvScene(model, maxgeom=4)
  for i, kind in enumerate(
    [mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ARROW]
  ):
    geom = scene.geoms[i]
    mujoco.mjv_initGeom(
      geom,
      kind,
      np.ones(3),
      np.zeros(3),
      np.eye(3).ravel(),
      np.array([1.0, 0.5, 0, 0.4]),
    )
    mujoco.mjv_connector(geom, kind, 0.01, np.array([1.0, 0, 0]), np.array([1.0, 0, 1]))
  scene.ngeom = 2
  batches = viewer.overlay_batches(scene)
  np.testing.assert_allclose(batches["sphere"][0], [[1, 0, 0], [1, 0, 1]])
  np.testing.assert_allclose(batches["cylinder"][0], [[1, 0, 0.5]])
  np.testing.assert_allclose(batches["arrow"][0], [[1, 0, 0]])
  np.testing.assert_allclose(batches["arrow"][2], [[0.01, 0.01, 1]])
  assert batches["arrow"][3][0, 3] == pytest.approx(0.4)


def test_web_server_does_not_mutate_plant_and_queues_controls(port):
  model, data = model_data()
  before_qpos, before_time = data.qpos.copy(), data.time
  before_scale = model.vis.scale.framewidth
  recording = SimpleNamespace(
    duration=2,
    times=[0, 0.02, 0.04, 2],
    events=[0, 1],
    index=lambda t: max(0, np.searchsorted([0, 0.02, 0.04, 2], t, side="right") - 1),
  )
  timeline = Timeline(recording, paused=True)
  with viewer.WebMujocoViewer(
    model, data, key_callback=timeline.key, port=port, wait=False
  ) as web:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    assert (
      opener.open(f"http://127.0.0.1:{web.server.get_port()}", timeout=5).status == 200
    )
    thread_ids = []
    thread = threading.Thread(
      target=lambda: web._actions.put(
        lambda: (thread_ids.append(threading.get_ident()), timeline.key(ord(".")))
      )
    )
    thread.start()
    thread.join()
    assert timeline.position == 0
    web.sync()
    assert timeline.position == 0.02
    assert thread_ids == [threading.get_ident()]
    web._seek.value = 0.8  # Browser state arrives before its callback is queued.
    web._last_sync = -float("inf")
    web.sync()
    assert web._seek.value == 0.8  # Never overwrite a pending seek with old time.
    assert timeline.position == 0.02
    web._actions.put(lambda: setattr(timeline, "position", 0.8))
    web._last_sync = -float("inf")
    web.sync()
    assert web._seek.value == timeline.position == 0.8
    commands = {handle.hotkey: handle for handle in web._hotkeys}
    assert {
      "space",
      ".",
      ",",
      "N",
      "P",
      "arrowleft",
      "arrowright",
      "home",
      "end",
    } <= commands.keys()
    commands["space"]._impl.trigger_cb[0](None)
    assert timeline.paused  # Keyboard callbacks also wait for the replay thread.
    web._last_sync = -float("inf")
    web.sync()
    assert not timeline.paused
    assert "播放中" in web._play_state.content
    assert web._play_button.label == "暂停播放 · Space"
    commands["."]._impl.trigger_cb[0](None)
    web.sync()
    assert timeline.paused and timeline.position == 2
    web._last_sync = -float("inf")
    web.sync()
    assert "已暂停" in web._play_state.content
    web._frame_count.value = 2
    web._queue_frame_step(-1)
    assert timeline.position == 2
    web.sync()
    assert timeline.position == 0.02 and timeline.paused
    commands["]"]._impl.trigger_cb[0](None)
    web._last_sync = -float("inf")
    web.sync()
    assert timeline.speed.rate == web._rate_input.value == 2
    assert "2×" in web._rate_display.content
    web._rate_input.value = 0.75
    web._last_sync = -float("inf")
    web.sync()
    assert web._rate_input.value == 0.75  # Preserve an edit pending dispatch.
    web._actions.put(lambda: timeline.speed.set_rate(0.75))
    web._last_sync = -float("inf")
    web.sync()
    assert web._published_rate == 0.75
    assert "0.75×" in web._rate_display.content
    np.testing.assert_array_equal(data.qpos, before_qpos)
    assert data.time == before_time
    assert model.vis.scale.framewidth == before_scale


def test_same_frozen_command_produces_identical_scores_and_traces_with_web(
  port, tmp_path, monkeypatch
):
  trial = Trial(0, 0, 0, (0.0, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1)
  config = dict(
    seed=42,
    grid=[1, 1],
    plane_x=0,
    y_range=[-1, 1],
    z_range=[0, 1],
    episode_seconds=0.2,
    display_hold_seconds=0,
    tolerances=dict(
      position_m=0.05, normal_degrees=15, velocity_mps=0.5, timing_seconds=0.04
    ),
  )

  class Agent:
    dt = 0.02
    accepted = True
    decision = dict(reason="accepted")
    translation = np.zeros(3)

    def __init__(self):
      m, d = model_data()
      self.robot = SimpleNamespace(model=m, data=d)
      self.reset()

    def reset(self):
      self.time = 0.0

    def advance(self, command=None, on_physics_step=None):
      assert on_physics_step is None
      self.time += self.dt
      self.robot.data.qpos[:3] = self.measure()["position"]
      mujoco.mj_forward(self.robot.model, self.robot.data)

    def measure(self):
      return dict(
        time=self.time,
        position=[self.time - 0.1, 0, 0.5],
        normal=[1, 0, 0],
        velocity=[1, 0, 0],
        phase="RETURN" if self.time >= 0.18 else "SWING",
        recovered=self.time >= 0.18,
        motion_index=0,
        tilt_degrees=0,
      )

    def lost_standing(self, sample):
      return False

  monkeypatch.setattr(viewer, "settings", viewer.ViewerSettings(port=port, wait=False))
  outcomes, traces = [], []
  for render in [False, True]:
    output = tmp_path / str(render)
    output.mkdir()
    env = MujocoCommandEnvironment(
      Agent(), config, output, render=render, command_only=True
    )
    try:
      outcomes.append(env.run_trial(trial, 0))
    finally:
      env.close()
    traces.append(json.loads((output / "traces/trial_00000.json").read_text()))
  assert outcomes[0] == outcomes[1]
  assert traces[0] == traces[1]
  assert outcomes[0]["target_pass"]


def test_native_backend_remains_explicit_and_does_not_construct_web(monkeypatch):
  import mujoco.viewer

  sentinel = object()
  monkeypatch.setattr(viewer, "settings", viewer.ViewerSettings(backend="native"))
  monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: sentinel)
  assert viewer.launch_passive(None, None) is sentinel


def test_planner_snapshot_has_independent_state_and_delivered_clock():
  class Core:
    def __init__(self):
      self.config = SimpleNamespace(x_hit=0.0)
      self.samples = []
      self.last_update_diagnostics = {}

    def update(self, stamp, position):
      self.samples.append((stamp, position.copy()))
      self.last_update_diagnostics = dict(samples=len(self.samples))
      return SimpleNamespace(
        t_strike=stamp + 0.3,
        p_intercept=np.array([0.0, -0.8, 0.3]),
        v_racket=np.array([2.0, 0, 1]),
        n_racket=np.array([1.0, 0, 0]),
        v_ball_outgoing=np.array([4.0, 0, 2]),
        target_land=np.array([2.4, -0.7, 0]),
        clears_net=True,
        bypasses_net_posts=False,
        valid=True,
        num_bounces=1,
        stage1_fit_rmse_m=0.001,
      )

  source = SimpleNamespace(
    planner=Core(),
    params={},
    latest=None,
    last_update_diagnostics={},
    lock=threading.Lock(),
  )
  first, second = PlannerSnapshot(source), PlannerSnapshot(source)
  first.update_observations([(10.0, [1.0, -0.8, 0.3], 100.0)])
  assert first.latest_command(100.0).deadline == pytest.approx(100.3)
  assert first.latest_command(100.3) is None
  assert not source.planner.samples and not second.planner.samples
  assert second.latest is None
  first.config.x_hit = -0.25
  assert second.config.x_hit == 0

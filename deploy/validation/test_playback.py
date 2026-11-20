"""Deterministic pacing, pause boundaries and independent slow-motion rendering."""

from types import SimpleNamespace

import numpy as np
import pytest

from deploy.playback import PlaybackClock, interpolate_qpos, run_viewer


@pytest.mark.parametrize("rate", (0.1, 0.25, 0.5, 1.0, 2.0))
def test_wall_clock_speed_preserves_fixed_control_period(rate):
  clock = PlaybackClock(0.02)
  steps = clock.advance(0, False, rate)
  for i in range(1, 10001):
    steps += clock.advance(i / 1000, False, rate)
  assert steps == round(10 * rate / 0.02)
  assert clock.dropped == 0


def test_pause_rate_change_and_overload_do_not_create_fast_forward_bursts():
  clock = PlaybackClock()
  clock.advance(0, False, 1)
  assert clock.advance(0.019, False, 1) == 0
  assert clock.advance(100, True, 1) == 0
  assert clock.advance(200, False, 1) == 0
  assert clock.advance(200.02, False, 1) == 1
  assert clock.advance(200.021, False, 0.25) == 0
  assert clock.advance(200.101, False, 0.25) == 1
  assert clock.advance(300, False, 0.25) <= 5
  assert clock.dropped > 0


def test_display_quaternion_interpolation_takes_short_path_and_preserves_inputs():
  before = np.zeros(36)
  before[3] = 1
  after = before.copy()
  after[:3], after[7:] = 2, 1
  after[3] = -1
  middle = interpolate_qpos(before, after, 0.5)
  np.testing.assert_array_equal(middle[:3], [1, 1, 1])
  np.testing.assert_array_equal(middle[7:], np.full(29, 0.5))
  np.testing.assert_array_equal(middle[3:7], [1, 0, 0, 0])
  assert before[3] == 1 and after[3] == -1


@pytest.mark.parametrize("rate", (0.1, 0.25, 1.0))
def test_slow_motion_does_not_reduce_display_frame_rate(monkeypatch, rate):
  wall = SimpleNamespace(now=0.0)
  monkeypatch.setattr("deploy.playback.time.monotonic", lambda: wall.now)
  monkeypatch.setattr(
    "deploy.playback.time.sleep",
    lambda seconds: setattr(wall, "now", wall.now + seconds),
  )
  session = SimpleNamespace(
    paused=False, world=SimpleNamespace(control_dt=0.02), steps=0
  )
  session.step = lambda: setattr(session, "steps", session.steps + 1)

  class Viewer:
    running = True
    single_step = False
    frames = SimpleNamespace(capture=lambda _session: None)

    def __init__(self):
      self.session, self.rate = session, rate
      self.rendered = []

    def drain(self, _directory):
      if wall.now >= 5:
        self.running = False

    def update(self, fraction):
      self.rendered.append((wall.now, fraction))

  viewer = Viewer()
  run_viewer(viewer, None)
  assert session.steps == pytest.approx(5 * rate / 0.02, abs=1)
  assert len(viewer.rendered) >= 290
  assert max(np.diff([t for t, _ in viewer.rendered])) < 0.02
  assert any(0 < fraction < 1 for _, fraction in viewer.rendered)

"""Missing-ball shading follows displayed sample age, without altering seeks."""

import json

import numpy as np
from deploy.tools.ball_visibility import BallVisibility, gaps_from_samples
from deploy.tools.real_viewer import ReplayLog


def test_gaps_include_edges_and_ignore_short_dropouts():
  coverage = BallVisibility()
  coverage.add(12, 12.1)
  coverage.add(12.05, 12.2)
  coverage.add(12.7, 13)
  coverage.add(15, 16)
  np.testing.assert_allclose(coverage.gaps(10, 18), [(0, 2), (3, 5), (6, 8)])
  assert BallVisibility().gaps(0, 8) == [(0, 8)]
  assert BallVisibility().gaps(0, 0) == []
  assert BallVisibility().gaps(0, 0.99) == []
  assert gaps_from_samples([0, 0.1, 0.2, 2], 3) == [(0.30000000000000004, 2.0)]


def test_replay_gaps_use_accepted_observations_and_sample_time(tmp_path):
  def ball(t, *, stamp=None, sample=None, position=None, frame="hope_table"):
    return dict(
      time=t,
      kind="mocap",
      data=dict(
        kind="ball",
        frame=frame,
        stamp=t if stamp is None else stamp,
        sample_monotonic=t if sample is None else sample,
        position=[1, 0, 1] if position is None else position,
      ),
    )

  events = [
    dict(time=100, kind="state", data={}),
    ball(102),
    ball(102.05),
    ball(102.5, stamp=102.05),  # Duplicate source stamp does not extend visibility.
    ball(103, sample=102),  # Already stale on arrival.
    ball(104, position=[float("nan"), 0, 0]),
    ball(105, frame="wrong"),
    ball(106),
    ball(106.05),
    dict(time=109, kind="state", data={}),
  ]
  (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
  log = ReplayLog(tmp_path)
  try:
    np.testing.assert_allclose(log.ball_gaps, [(0, 2), (2.15, 6), (6.15, 9)])
    expected = log.ball_gaps.copy()
    log.seek(8)
    log.seek(1)
    assert log.ball_gaps == expected
    with (tmp_path / "events.jsonl").open("a") as f:
      f.write(json.dumps(ball(115)) + "\n")
    assert log.duration == 9  # Snapshot boundaries do not shift during seeks.
  finally:
    log.close()
  reopened = ReplayLog(tmp_path)
  try:
    assert reopened.duration == 15
    assert reopened.ball_gaps[-1][1] == 15
  finally:
    reopened.close()

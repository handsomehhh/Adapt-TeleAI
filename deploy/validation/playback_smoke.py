"""Bounded real-model wall-clock and display-rate checks; CPU and localhost only."""

import json
import time
from pathlib import Path

import torch

from deploy import ROOT
from deploy.artifacts import Artifacts
from deploy.playback import run_viewer
from deploy.session import Session
from deploy.viewer import Viewer


class ProbeViewer(Viewer):
  def __init__(self, session, port, seconds):
    super().__init__(session, port=port)
    self.start = time.monotonic()
    self.seconds = seconds
    self.draw_times = []

  def drain(self, output):
    super().drain(output)
    if time.monotonic() - self.start >= self.seconds:
      self.running = False

  def update(self, fraction=1.0):
    super().update(fraction)
    if hasattr(self, "draw_times"):
      self.draw_times.append(time.monotonic())


def check(output, rate, port, seconds=6):
  artifacts = Artifacts()
  originals = (
    sorted((artifacts.run / "snapshot/dataset/g1_stage1/motions").glob("*.npz"))
    if artifacts.run
    else []
  )
  session = Session(
    artifacts, originals[0] if originals else artifacts.motion, "tracker"
  )
  session.tracker_loop = True
  viewer = ProbeViewer(session, port, seconds)
  viewer.rate, session.paused = rate, False
  try:
    run_viewer(viewer, output)
    span = time.monotonic() - viewer.start
    factor = len(session.records) * session.world.control_dt / span
    fps = len(viewer.draw_times) / span
    result = {
      "requested_rate": rate,
      "wall_seconds": span,
      "physics_dt": session.world.dt,
      "control_dt": session.world.control_dt,
      "steps": len(session.records),
      "server_fps": fps,
      "realtime_factor": factor,
      "relative_speed_error": abs(factor - rate) / rate,
      "scope": "native CPU with actual Tracker; server sends, not browser draw FPS",
    }
    assert result["relative_speed_error"] < 0.15, result
    assert fps >= 40, result
    session.save(output)
    return result
  finally:
    viewer.close()


def main():
  torch.set_num_threads(2)
  root = ROOT / "deploy/recordings" / time.strftime("pacing_validation_%Y%m%d_%H%M%S")
  Path(root).mkdir(parents=True)
  results = [
    check(root / "realtime", 1.0, 18090),
    check(root / "slow_motion", 0.25, 18091),
  ]
  (root / "pacing_checks.json").write_text(json.dumps(results, indent=2) + "\n")
  print(json.dumps({"results": results, "output": str(root)}, indent=2))


if __name__ == "__main__":
  main()

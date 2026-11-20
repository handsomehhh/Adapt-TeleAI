"""Launch per-module headless tests or the interactive CPU deployment viewer."""

import argparse
import json
import os
import signal
from datetime import datetime
from pathlib import Path

# This platform is intentionally CPU-only; never create a training GPU context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from deploy import ROOT
from deploy.artifacts import Artifacts
from deploy.control import MODES
from deploy.playback import run_viewer
from deploy.replay import Replay
from deploy.session import Session


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path)
  parser.add_argument("--run", type=Path)
  parser.add_argument("--bundle", type=Path)
  parser.add_argument("--motion", type=Path)
  parser.add_argument("--mode", choices=(*MODES, "replay"), default="hierarchy")
  parser.add_argument("--viewer", choices=("web", "headless"), default="web")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8085)
  parser.add_argument("--steps", type=int, default=0)
  parser.add_argument("--max-record-steps", type=int, default=0)
  parser.add_argument("--record-chunk-steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--scenario", choices=("independent", "continuous"), default="continuous"
  )
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--noise", type=float, default=0.005)
  parser.add_argument("--dropout", type=float, default=0.05)
  parser.add_argument("--delay-steps", type=int, default=2)
  parser.add_argument("--trace", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--suite", action="store_true")
  for role in ("tracker", "mvae", "rally", "standing"):
    parser.add_argument("--" + role, type=Path)
  selected, _ = parser.parse_known_args()
  if selected.config:
    settings = json.loads(selected.config.read_text())
    allowed = {action.dest for action in parser._actions} - {"help", "config"}
    if set(settings) - allowed:
      parser.error(f"Unknown configuration fields: {set(settings) - allowed}")
    for key in (
      "run",
      "bundle",
      "motion",
      "output",
      "trace",
      "tracker",
      "mvae",
      "rally",
      "standing",
    ):
      if settings.get(key):
        settings[key] = (ROOT / settings[key]).resolve()
    parser.set_defaults(**settings)
  args = parser.parse_args()
  print(f"[deploy] 运行库已加载：mode={args.mode}, viewer={args.viewer}", flush=True)
  if (
    args.steps < 0
    or args.max_record_steps < 0
    or args.record_chunk_steps < 1
    or not 0.5 <= args.speed <= 2
    or args.noise < 0
    or not 0 <= args.dropout <= 1
    or args.delay_steps < 0
  ):
    parser.error("Invalid steps/speed/perception settings")
  torch.set_num_threads(2)
  output = args.output or ROOT / "deploy/recordings" / datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
  )
  if output.exists():
    parser.error("Choose a new output directory to preserve existing recordings")
  output.mkdir(parents=True)
  artifacts = None
  if args.mode != "replay":
    print("[deploy] 正在定位模型与参考动作……", flush=True)
    artifacts = Artifacts(
      args.run,
      args.bundle,
      {role: getattr(args, role) for role in ("tracker", "mvae", "rally", "standing")},
    )
    motion = args.motion or artifacts.motion
    if args.mode == "tracker" and not args.motion and artifacts.run and not args.suite:
      originals = sorted(
        (artifacts.run / "snapshot/dataset/g1_stage1/motions").glob("*.npz")
      )
      if originals:
        motion = originals[0]
    if motion is None:
      parser.error(
        "Specify --motion with a standalone bundle or a run without corrected motions"
      )
  sessions = []
  viewer = None
  try:
    for mode in MODES if args.suite else (args.mode,):
      if mode == "replay":
        if not args.trace:
          parser.error("Replay requires --trace")
        print(f"[deploy] 正在加载回放：{args.trace}", flush=True)
        session = Replay(args.trace)
      else:
        print(f"[deploy] 正在构建 CPU 仿真并加载 {mode} 模型……", flush=True)
        session = Session(
          artifacts,
          motion,
          mode,
          args.seed,
          args.scenario,
          args.noise,
          args.dropout,
          args.delay_steps,
          random_initial_motion=not args.motion,
        )
        session.speed = args.speed
        session.max_record_steps = args.max_record_steps
      directory = output / mode if args.suite else output
      directory.mkdir(parents=True, exist_ok=True)
      if isinstance(session, Session):
        session.records.directory = directory.resolve()
        session.records.chunk_steps = args.record_chunk_steps
        session.save(directory, completed=False)
      sessions.append((session, directory))
      if args.viewer == "headless" or args.suite:
        remaining = args.steps or (250 if args.suite else 0)
        count = 0
        while remaining == 0 or count < remaining:
          session.step()
          count += 1
          if session.paused:
            break
      else:
        from deploy.viewer import Viewer

        print(f"[deploy] 正在启动网页：{args.host}:{args.port}", flush=True)
        viewer = Viewer(session, args.host, args.port)
        print(
          f"Open http://{args.host}:{viewer.server.get_port()} (starts paused)",
          flush=True,
        )
        run_viewer(viewer, directory, args.steps)
      session.save(directory)
      print(
        json.dumps(
          {
            "mode": mode,
            "output": str(directory),
            "steps": getattr(session.records, "total", len(session.records)),
            "trials": len(session.trials),
          }
        ),
        flush=True,
      )
  except KeyboardInterrupt:
    pass
  except BaseException as error:
    for session, _ in sessions:
      session.error = str(error)
    (output / "error.json").write_text(
      json.dumps({"error": str(error), "completed": False}, indent=2)
    )
    raise
  finally:
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
      for session, directory in sessions:
        session.save(directory)
    finally:
      try:
        if viewer:
          viewer.close()
      finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
  main()

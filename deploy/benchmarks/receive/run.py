"""Stage two: independent random serves through the real planner interface."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from deploy.benchmarks.receive.feeding import generate_suite, load_suite


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--config", type=Path, default=Path(__file__).with_name("center.yaml"))
  p.add_argument("--suite", type=Path)
  p.add_argument("--seed", type=int)
  p.add_argument("--balls-per-cell", type=int)
  p.add_argument("--generate-only", action="store_true")
  p.add_argument("--limit", type=int)
  p.add_argument("--render", action="store_true")
  p.add_argument("--playback-rate", type=float, default=1)
  p.add_argument("--capture-first", type=int, default=0)
  p.add_argument(
    "--return-target",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    help="Explicit task landing goal; preserves A3 arithmetic/physics and frozen incoming balls",
  )
  p.add_argument("--output", type=Path)
  for name, file in [
    ("task", "task.yaml"),
    ("rally", "rally_mujoco.yaml"),
    ("runtime", "real.yaml"),
    ("mocap", "mocap.yaml"),
  ]:
    p.add_argument(
      f"--{name}-config", type=Path, default=Path("deploy/hope_g1/configs") / file
    )
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(p)
  args = p.parse_args()
  configure_viewer(args)
  if (
    args.playback_rate <= 0
    or args.capture_first < 0
    or (args.limit is not None and args.limit < 1)
  ):
    p.error("Invalid playback-rate/capture-first/limit")
  if args.suite and (args.seed is not None or args.balls_per_cell is not None):
    p.error("Frozen suite cannot override sampling")
  output = args.output or Path("deploy/recordings/receive") / datetime.now().strftime(
    "center_%Y%m%d_%H%M%S"
  )
  output.mkdir(parents=True, exist_ok=True)
  if (output / "suite.json").exists():
    p.error("Use a new output directory")
  torch.set_num_threads(1)
  rally = yaml.safe_load(args.rally_config.read_text())
  if args.suite:
    suite = json.loads(args.suite.read_text())
    serves = load_suite(suite)
    config = suite["config"]
    if suite["physics"] != rally["ball"]:
      p.error("Rally configuration differs from frozen ball physics")
  else:
    config = yaml.safe_load(args.config.read_text())
    for key in ["seed", "balls_per_cell"]:
      if getattr(args, key) is not None:
        config[key] = getattr(args, key)
    suite = generate_suite(config, rally["ball"])
    serves = suite["serves"]
  config = {**config, "physical_ball": suite["physics"]}
  (output / "suite.json").write_text(json.dumps(suite, indent=2))
  print(
    f"Frozen {len(serves)} independent central serves: {output / 'suite.json'}",
    flush=True,
  )
  if args.generate_only:
    return
  from deploy.benchmarks.receive.environment import ReceiveEnvironment
  from deploy.benchmarks.receive.reporting import write_report
  from deploy.hope_g1.receive_adapter import HopeReceiveAdapter

  agent = HopeReceiveAdapter(
    args.task_config,
    args.rally_config,
    args.runtime_config,
    args.mocap_config,
    output,
    plane_x=config["plane_x"],
    return_target=args.return_target,
  )
  estimator = yaml.safe_load(args.mocap_config.read_text())["estimator"]
  metadata = {
    **agent.metadata,
    "evaluation_mode": "planner_receive",
    "observation_hz": config["observation_hz"],
    "position_noise_std": config["position_noise_std"],
    "estimator": estimator,
    "serves_conditioned_on": "Incoming geometry only; oracle crossing never enters agent",
    "planner_privileged_ball_velocity": False,
    "rally_config": str(args.rally_config),
  }
  (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
  env = ReceiveEnvironment(
    agent,
    config,
    output,
    estimator,
    render=args.render,
    playback_rate=args.playback_rate,
    capture_first=args.capture_first,
  )
  status = "running"
  try:
    with (output / "results.jsonl").open("w") as stream:
      for i, feed in enumerate(serves[: args.limit]):
        row = env.run_ball(feed, i)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        print(
          f"[{i + 1}/{len(serves)}] accepted={row['accepted']} motion={row['motion_index']} hit={row['ball_hit']} legal_return={row['legal_return']} error={row['failure']}",
          flush=True,
        )
        if env.quit:
          status = "interrupted"
          break
      else:
        status = "complete" if len(env.completed) == len(serves) else "partial"
  except KeyboardInterrupt:
    status = "interrupted"
  except Exception:
    status = "error"
    raise
  finally:
    env.close()
    report = write_report(output, suite, env.completed, metadata, status)
  print(json.dumps(report["summary"], indent=2))
  print(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
  main()

"""Generate independent command suites, execute MuJoCo, and export reports."""

import argparse
import importlib
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from deploy.benchmarks.racket_command.domain import resolve
from deploy.benchmarks.racket_command.sampling import (
  load_suite,
  sample_suite,
  suite_document,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path(__file__).with_name("default.yaml")
  )
  parser.add_argument(
    "--suite", type=Path, help="Replay the exact pre-generated command suite"
  )
  parser.add_argument("--generate-only", action="store_true")
  parser.add_argument("--seed", type=int)
  parser.add_argument(
    "--domain", type=Path, help="Use an exported training command support JSON"
  )
  parser.add_argument(
    "--unrestricted",
    action="store_true",
    help="Explicit legacy rectangular stress domain",
  )
  parser.add_argument("--samples-per-cell", type=int)
  parser.add_argument(
    "--point",
    type=float,
    nargs=2,
    metavar=("Y", "Z"),
    help="Test one exact point on the fixed X plane",
  )
  parser.add_argument("--directions", type=int)
  parser.add_argument("--speed-levels", type=int)
  parser.add_argument("--tts-levels", type=int)
  parser.add_argument(
    "--tts-values",
    type=float,
    nargs="+",
    help="Explicit TTS seconds; samples-per-cell targets repeated at each TTS",
  )
  parser.add_argument("--position-sampling", choices=["fixed_per_cell", "random"])
  parser.add_argument(
    "--limit", type=int, help="Run only a saved prefix, mark report partial"
  )
  parser.add_argument("--render", action="store_true")
  parser.add_argument(
    "--hide-ball",
    action="store_true",
    help="Hide the physical ball without disabling dynamics",
  )
  parser.add_argument(
    "--command-only",
    action="store_true",
    help="Disable ball physics for pure command tracking",
  )
  parser.add_argument("--playback-rate", type=float, default=1)
  parser.add_argument("--capture-first", type=int, default=0)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--adapter", default="deploy.hope_g1.command_adapter:HopeCommandAdapter"
  )
  parser.add_argument(
    "--task-config", type=Path, default=Path("deploy/hope_g1/configs/task.yaml")
  )
  parser.add_argument(
    "--rally-config",
    type=Path,
    default=Path("deploy/hope_g1/configs/rally_mujoco.yaml"),
  )
  parser.add_argument(
    "--runtime-config", type=Path, default=Path("deploy/hope_g1/configs/real.yaml")
  )
  parser.add_argument(
    "--mocap-config", type=Path, default=Path("deploy/hope_g1/configs/mocap.yaml")
  )
  parser.add_argument("--home-index", type=int, default=0)
  parser.add_argument("--motion-allow", default="all")
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(parser)
  args = parser.parse_args()
  configure_viewer(args)
  if (
    args.playback_rate <= 0
    or args.capture_first < 0
    or (args.limit is not None and args.limit < 1)
  ):
    parser.error("Positive playback-rate/limit and nonnegative capture-first required")
  if args.suite and any(
    v is not None
    for v in (
      args.seed,
      args.samples_per_cell,
      args.domain,
      args.point,
      args.directions,
      args.speed_levels,
      args.tts_levels,
      args.tts_values,
      args.position_sampling,
    )
  ):
    parser.error("Suite replay does not allow sampling overrides")
  if args.unrestricted and (args.suite or args.domain):
    parser.error("unrestricted cannot be combined with suite/domain")
  output = args.output or Path(
    "deploy/recordings/racket_command"
  ) / datetime.now().strftime("%Y%m%d_%H%M%S")
  output.mkdir(parents=True, exist_ok=True)
  if (output / "results.jsonl").exists() or (output / "suite.json").exists():
    parser.error("Use a new output directory to preserve existing evidence")
  if args.suite:
    document = json.loads(args.suite.read_text())
    trials = load_suite(document)
    config = deepcopy(document["config"])
    # Old suites predate A3's lateral net overhang; preserve their physics.
    if "physical_ball" in config:
      config["physical_ball"].setdefault("net_overhang", 0.0)
  else:
    config = yaml.safe_load(args.config.read_text())
    if args.unrestricted:
      config.pop("domain_file", None)
      config.pop("coverage_domain", None)
    elif args.domain:
      config["domain_file"] = str(args.domain)
    for key in ("seed", "samples_per_cell"):
      if getattr(args, key) is not None:
        config[key] = getattr(args, key)
    if args.position_sampling:
      config["position_sampling"] = args.position_sampling
      if args.position_sampling == "random":
        config.pop("sweep", None)
    if args.point:
      if args.position_sampling == "random":
        parser.error("point cannot be combined with random position sampling")
      config["fixed_point"] = args.point
      config["position_sampling"] = "fixed_per_cell"
    axes = (args.directions, args.speed_levels, args.tts_levels)
    if any(v is not None for v in axes):
      if config.get("position_sampling") == "random":
        parser.error("Factorial sweeps require fixed_per_cell positions")
      sweep = config.setdefault(
        "sweep", dict(directions=5, speed_levels=2, tts_levels=2)
      )
      for key, value in zip(
        ("directions", "speed_levels", "tts_levels"), axes, strict=True
      ):
        if value is not None:
          sweep[key] = value
      count = sweep["directions"] * sweep["speed_levels"] * sweep["tts_levels"]
      if args.samples_per_cell is not None and args.samples_per_cell != count:
        parser.error("samples-per-cell must match the sweep product")
      config["samples_per_cell"] = count
    elif args.samples_per_cell is not None:
      # Explicit N keeps N random vector/time variants at the same spatial point.
      config.pop("sweep", None)
    if args.tts_values is not None:
      if any(v is not None for v in axes):
        parser.error("tts-values cannot be combined with factorial sweep options")
      config.pop("sweep", None)
      config["tts_values"] = args.tts_values
    config = resolve(config)
    trials = sample_suite(config)
    document = suite_document(config, trials)
  # Saved BEFORE loading weights or the motion library; no acceptance resampling.
  (output / "suite.json").write_text(json.dumps(document, indent=2))
  if config.get("rally_sampling_audit") is not None:
    (output / "sampling_audit.json").write_text(
      json.dumps(config["rally_sampling_audit"], indent=2)
    )
  print(
    f"Saved {len(trials)} independent commands: {output / 'suite.json'}", flush=True
  )
  if args.generate_only:
    return
  mode = config.get("evaluation_mode", "command_ball_diagnostic")
  if mode not in ("policy_tracking", "command_ball_diagnostic"):
    parser.error("Unknown evaluation_mode in test configuration")
  command_only = args.command_only or mode == "policy_tracking"
  config["evaluation_mode"] = (
    "policy_tracking" if command_only else "command_ball_diagnostic"
  )
  import torch

  torch.set_num_threads(1)
  from deploy.benchmarks.racket_command.environment import MujocoCommandEnvironment
  from deploy.benchmarks.racket_command.visualization import write_reports

  module, name = args.adapter.split(":")
  factory = getattr(importlib.import_module(module), name)
  allow = (
    None
    if args.motion_allow == "all"
    else [int(v) for v in args.motion_allow.split(",")]
  )
  agent = factory(
    args.task_config,
    args.rally_config,
    args.runtime_config,
    args.mocap_config,
    output,
    home_index=args.home_index,
    allow=allow,
  )
  metadata = {
    **agent.metadata,
    "adapter": args.adapter,
    "evaluation_mode": "policy_tracking" if command_only else "command_ball_diagnostic",
    "planner_used": False,
    "selector_used": True,
    "evaluation_scope": "Existing motion selector + actor tracking of independently sampled commands",
    "rally_constraints_enabled": config.get("rally_constraints", {}).get(
      "enabled", False
    ),
    "normal_metric": "Unoriented disk normal (equivalent +/- faces)",
    "timing_metric": "Closest position within +/- .25s, diagnostic only; tracking scored at exact deadline",
    "reset_count_per_trial": 1,
    "ball_dynamics": "disabled"
    if command_only
    else "A3 ball/table parameters with ballistic/table/net/moving-racket collision kernels at physics substeps; no outgoing steering",
    "display_ball": not args.hide_ball and not command_only,
  }
  (output / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2))
  env = MujocoCommandEnvironment(
    agent,
    config,
    output,
    args.render,
    args.playback_rate,
    args.capture_first,
    show_ball=not args.hide_ball and not command_only,
    command_only=command_only,
  )
  status = "running"
  try:
    with (output / "results.jsonl").open("w") as stream:
      for i, trial in enumerate(trials[: args.limit]):
        result = env.run_trial(trial, i)
        stream.write(
          json.dumps({**result, "command": asdict(trial)}, allow_nan=False) + "\n"
        )
        stream.flush()
        print(
          f"[{i + 1}/{len(trials)}] id={trial.trial_id} motion={result['motion_index']} "
          f"accepted={result['accepted']} success={result['success']} "
          f"pos={result.get('position_error_m')} "
          f"reason={(result.get('decision') or {}).get('reason')}",
          flush=True,
        )
        if env.quit:
          status = "interrupted"
          break
      else:
        status = "complete" if len(env.completed) == len(trials) else "partial"
  except KeyboardInterrupt:
    status = "interrupted"
  except Exception:
    status = "error"
    raise
  finally:
    env.close()
    report = write_reports(output, config, document, env.completed, metadata, status)
  summary = dict(tracking=report["summary"])
  if not command_only:
    summary["ball_physics"] = report["ball_summary"]
  print(json.dumps(summary, indent=2))
  print(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
  main()

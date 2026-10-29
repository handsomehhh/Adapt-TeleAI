"""104-D G1 hitter task entry (check / MuJoCo / real) on the shared runtime."""

import argparse
import json
import signal
from datetime import datetime
from pathlib import Path

import torch
import yaml

from deploy.hope_g1.deploy_agent import request_shutdown
from deploy.hope_g1_hitter.configuration import TASK, load_deployment
from deploy.hope_g1_hitter.controller import HitterController

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
HOPE_CONFIG_DIR = CONFIG_DIR.parent.parent / "hope_g1/configs"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env", choices=["check", "mujoco", "real"], default="check")
  parser.add_argument("--config", type=Path, default=CONFIG_DIR / "task.yaml")
  parser.add_argument("--artifact-dir", type=Path)
  parser.add_argument("--real-config", type=Path, default=CONFIG_DIR / "real.yaml")
  parser.add_argument(
    "--mocap-config", type=Path, default=HOPE_CONFIG_DIR / "mocap.yaml"
  )
  parser.add_argument(
    "--rally-config",
    type=Path,
    help="Default: rally_mujoco.yaml for MuJoCo, rally.yaml for real/check",
  )
  parser.add_argument("--duration", type=float, default=120)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--network-interface")
  parser.add_argument("--enable-motors", action="store_true")
  parser.add_argument(
    "--side-selection", choices=["auto", "forehand", "backhand"],
    help="Override hitter.side_selection in the rally config",
  )
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--sim-controls", choices=["auto", "manual"], default="auto")
  parser.add_argument(
    "--sim-startup", choices=["ground", "suspended"], default="suspended"
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--min-hit-rate", type=float)
  parser.add_argument("--min-return-rate", type=float)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--torch-threads", type=int, default=1)
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(parser)
  args = parser.parse_args()
  configure_viewer(args)
  if args.duration <= 0 or args.episodes < 1 or args.torch_threads < 1:
    parser.error("duration, episodes and torch-threads must be positive")
  if args.env != "real" and args.enable_motors:
    parser.error("--enable-motors is only valid for --env real")
  if args.env == "real" and args.render:
    parser.error("render is a simulation option")
  for rate in (args.min_hit_rate, args.min_return_rate):
    if rate is not None and (args.env != "mujoco" or not 0 <= rate <= 1):
      parser.error("min-hit/return-rate must be in [0, 1] and require MuJoCo")
  signal.signal(signal.SIGTERM, request_shutdown)
  torch.set_num_threads(args.torch_threads)
  config = yaml.safe_load(args.config.read_text())
  if config.pop("task", None) != TASK:
    parser.error(f"This entry only accepts {TASK} task configs")
  artifact_dir = args.artifact_dir or Path(config["artifact_dir"])
  args.rally_config = args.rally_config or CONFIG_DIR / (
    "rally_mujoco.yaml" if args.env == "mujoco" else "rally.yaml"
  )
  options = yaml.safe_load(args.rally_config.read_text())
  if args.side_selection is not None:
    options["hitter"]["side_selection"] = args.side_selection
  options["hitter"]["planner_config"] = str(
    Path(options["hitter"]["planner_config"]).resolve()
  )
  if yaml.safe_load(args.mocap_config.read_text()).get("world_frame") != "hope_table":
    parser.error("HOPE G1 mocap must publish normalized packets in hope_table world")
  if args.env == "real" and options.get("physics_profile") != "real_a3":
    parser.error("Real HOPE requires the real_a3 profile (configs/rally.yaml)")
  deployment = load_deployment(
    artifact_dir, ball_config=options["ball"], world_config=options["world"]
  )
  runtime = yaml.safe_load(args.real_config.read_text())
  validated = HitterController(deployment, runtime, options)
  report = deployment.report()
  report.update(
    station=options["hitter"]["station"],
    side_selection=validated.side_selection,
    common_hit_plane_x_range=validated.geometry.common_x_range().tolist(),
    planner_config=options["hitter"]["planner_config"],
  )
  print(json.dumps(report, indent=2))
  print(validated.describe())
  if args.env == "check":
    return
  output = (
    args.output
    or Path("deploy/recordings/hope_g1_hitter")
    / f"{args.env}_{datetime.now():%Y%m%d_%H%M%S}"
  )
  resolved_rally = (
    output.with_suffix(".rally.yaml")
    if output.suffix == ".jsonl"
    else output / "rally_config.yaml"
  )
  resolved_rally.parent.mkdir(parents=True, exist_ok=True)
  resolved_rally.write_text(yaml.safe_dump(options, sort_keys=False))
  common = dict(
    controller_factory=HitterController,
    motion_index=0,
    motion_select="fixed",
    motion_allow=[0],
    duration=args.duration,
    output=output,
    simulation_config=resolved_rally,
  )
  if args.env == "mujoco":
    from deploy.g1_pingpong.simulation_runtime import run_deployment
    from deploy.hope_g1.random_feeder import RandomLaunchFeeder

    if options["feed"].get("mode") != "random_launch":
      parser.error("The hitter MuJoCo entry supports only feed.mode=random_launch")
    result = run_deployment(
      deployment,
      args.real_config,
      args.mocap_config,
      episodes=args.episodes,
      seed=args.seed,
      render=args.render,
      controls=args.sim_controls,
      startup=args.sim_startup,
      standing_only=False,
      ball_source="sim",
      minimum_hit_rate=args.min_hit_rate,
      minimum_return_rate=args.min_return_rate,
      feeder_factory=RandomLaunchFeeder,
      **common,
    )
    keys = (
      "passed",
      "flow_passed",
      "simulated_hits",
      "simulated_balls",
      "hit_rate",
      "simulated_legal_returns",
      "legal_return_rate",
      "error",
      "completed_swings",
      "recovered_swings",
    )
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))
    if not result["passed"]:
      raise SystemExit(1)
  else:
    from deploy.backends.sdk import network_interfaces, require_sdk
    from deploy.backends.unitree import run_control_loop
    from deploy.tools.preflight import check_configuration

    require_sdk()
    runtime, _ = check_configuration(
      args.real_config, args.mocap_config, args.network_interface
    )
    if runtime["network_interface"] not in network_interfaces():
      parser.error("Configured network interface does not exist")
    run_control_loop(
      deployment,
      runtime,
      args.mocap_config,
      enable_motors=args.enable_motors,
      **common,
    )


if __name__ == "__main__":
  main()

"""Independent HOPEPingPongMultiMotionG1 task entry (check / MuJoCo / real)."""

import argparse
import hashlib
import json
import signal
from datetime import datetime
from pathlib import Path

import torch
import yaml

from deploy.hope_g1.configuration import TASK, load_deployment
from deploy.hope_g1.controller import HopeController

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def request_shutdown(*_):
  raise KeyboardInterrupt("HOPE deployment shutdown requested")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env", choices=["check", "mujoco", "real"], default="check")
  parser.add_argument(
    "--config",
    type=Path,
    help="Default: local task.yaml for check/MuJoCo; task_real.yaml for real",
  )
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--env-config", type=Path)
  parser.add_argument("--agent-config", type=Path)
  parser.add_argument("--motion-dir", type=Path)
  parser.add_argument("--real-config", type=Path, default=CONFIG_DIR / "real.yaml")
  parser.add_argument("--mocap-config", type=Path, default=CONFIG_DIR / "mocap.yaml")
  parser.add_argument(
    "--rally-config",
    type=Path,
    help="Default: rally_mujoco_random.yaml for MuJoCo, rally.yaml for real/check",
  )
  parser.add_argument("--motion-index", type=int, default=0)
  parser.add_argument("--motion-select", choices=["fixed", "auto"], default="fixed")
  parser.add_argument("--motion-allow", default="all")
  parser.add_argument(
    "--target-source",
    choices=["hope_planner_optimized_2", "a3_planner", "reference"],
  )
  parser.add_argument("--a3-target-mode", choices=["raw", "envelope"])
  parser.add_argument("--duration", type=float, default=120)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--network-interface")
  parser.add_argument("--enable-motors", action="store_true")
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--sim-controls", choices=["auto", "manual"], default="auto")
  parser.add_argument(
    "--sim-startup", choices=["ground", "suspended"], default="suspended"
  )
  parser.add_argument("--standing-only", action="store_true")
  parser.add_argument("--ball-source", choices=["sim", "mocap"])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--min-hit-rate",
    type=float,
    help="Require this physical contact fraction in automatic MuJoCo receiving",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--min-return-rate",
    type=float,
    help="Require this fraction to clear the net and first bounce on the opponent table",
  )
  parser.add_argument("--torch-threads", type=int, default=1)
  args = parser.parse_args()
  if args.duration <= 0 or args.episodes < 1 or args.torch_threads < 1:
    parser.error("duration, episodes and torch-threads must be positive")
  if args.env != "real" and args.enable_motors:
    parser.error("--enable-motors is only valid for --env real")
  if args.env == "real" and args.ball_source == "sim":
    parser.error("Real HOPE deployment requires measured mocap")
  if args.env == "real" and (args.render or args.standing_only):
    parser.error("render/standing-only are simulation options")
  if args.min_hit_rate is not None and (
    args.env != "mujoco"
    or args.sim_controls != "auto"
    or args.standing_only
    or args.ball_source == "mocap"
    or not 0 <= args.min_hit_rate <= 1
  ):
    parser.error(
      "min-hit-rate must be in [0, 1] and requires automatic MuJoCo receiving"
    )
  if args.min_return_rate is not None and (
    args.env != "mujoco"
    or args.sim_controls != "auto"
    or args.standing_only
    or args.ball_source == "mocap"
    or not 0 <= args.min_return_rate <= 1
  ):
    parser.error(
      "min-return-rate must be in [0, 1] and requires automatic MuJoCo receiving"
    )
  signal.signal(signal.SIGTERM, request_shutdown)
  torch.set_num_threads(args.torch_threads)
  config_path = args.config or CONFIG_DIR / (
    "task_real.yaml" if args.env == "real" else "task.yaml"
  )
  config = yaml.safe_load(config_path.read_text())
  if config.pop("task", None) != TASK:
    parser.error("This entry only accepts HOPEPingPongMultiMotionG1 task configs")
  for name in ("checkpoint", "motion_dir", "env_config", "agent_config"):
    if getattr(args, name) is not None:
      config[name] = getattr(args, name)
  args.rally_config = args.rally_config or CONFIG_DIR / (
    "rally_mujoco_random.yaml" if args.env == "mujoco" else "rally.yaml"
  )
  options = yaml.safe_load(args.rally_config.read_text())
  planner_config = Path(
    options.get(
      "hope_planner_config", CONFIG_DIR / "hope_planner_optimized_2.yaml"
    )
  )
  options["hope_planner_config"] = str(planner_config.resolve())
  if yaml.safe_load(args.mocap_config.read_text()).get("world_frame") != "hope_table":
    parser.error("HOPE G1 mocap must publish normalized packets in hope_table world")
  if args.env == "real" and options.get("physics_profile") != "real_a3":
    parser.error("Real HOPE requires the real_a3 profile (configs/rally.yaml)")
  if args.target_source is not None:
    options["target_source"] = args.target_source
  if args.a3_target_mode is not None:
    options.setdefault("a3", {})["target_mode"] = args.a3_target_mode
  deployment = load_deployment(
    **config, ball_config=options["ball"], world_config=options["world"]
  )
  n = len(deployment.motions)
  try:
    allow = (
      list(range(n))
      if args.motion_allow == "all"
      else [int(i) for i in args.motion_allow.split(",")]
    )
    if (
      not 0 <= args.motion_index < n
      or not allow
      or len(set(allow)) != len(allow)
      or any(not 0 <= i < n for i in allow)
    ):
      raise ValueError
  except ValueError:
    parser.error(f"Motion IDs must be unique and within 0..{n - 1}")
  # Construction validates the controller contract before any transport opens.
  runtime = yaml.safe_load(args.real_config.read_text())
  validated_controller = HopeController(
    deployment,
    runtime,
    options,
    home_index=args.motion_index,
    allow=allow,
    selection=args.motion_select,
  )
  report = deployment.report()
  report.update(
    target_source=options.get("target_source", "reference"),
    planner_transport=options.get("planner_transport", "embedded"),
    target_mode=options.get("a3", {}).get("target_mode", "raw"),
    target_perturb_scale=options.get("target_perturb_scale"),
    return_steps=validated_controller.return_frame_count,
    select_settle_ticks=validated_controller.select_settle_ticks,
  )
  if options.get("target_source") in (
    "hope_planner_optimized_2",
    "a3_planner",
  ):
    planner_yaml = yaml.safe_load(planner_config.read_text())[
      "hope_planner_optimized_2"
    ]["ros__parameters"]
    report["planner"] = {
      "implementation": "hope_planner_optimized_2_A30808",
      "transport": options.get("planner_transport", "embedded"),
      "config_path": str(planner_config),
      "config_sha256": hashlib.sha256(planner_config.read_bytes()).hexdigest(),
      "ros_parameters": planner_yaml,
      "upstream_core_defaults": {
        "poly_order": 2,
        "fit_window": 30,
        "min_fit_samples": 10,
        "mocap_hz": 180.0,
        "clear_x_max_m": 2.2,
        "max_predict_time_s": 2.0,
        "bounce_z_tol_m": 0.02,
        "racket_radius_m": 0.075,
        "ball_radius_m": 0.02,
        "ball_mass_kg": 0.0027,
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "table_length_m": 2.74,
        "table_width_m": 1.525,
        "table_height_m": 0.76,
        "net_x_m": 1.37,
        "net_height_m": 0.1525,
        "net_overhang_m": 0.15,
      },
      "g1_hit_plane_x_m": planner_yaml["x_hit"],
      "parameter_overrides": {
        "x_hit": planner_yaml["x_hit"],
        "publish_intercept_x_min": planner_yaml["publish_intercept_x_min"],
        "publish_intercept_x_max": planner_yaml["publish_intercept_x_max"],
      },
    }
  print(json.dumps(report, indent=2))
  if args.env == "check":
    return
  output = (
    args.output
    or Path("deploy/recordings/hope_g1") / f"{args.env}_{datetime.now():%Y%m%d_%H%M%S}"
  )
  # Record and consume the same resolved settings, including CLI overrides.
  resolved_rally = (
    output.with_suffix(".rally.yaml")
    if output.suffix == ".jsonl"
    else output / "rally_config.yaml"
  )
  resolved_rally.parent.mkdir(parents=True, exist_ok=True)
  resolved_rally.write_text(yaml.safe_dump(options, sort_keys=False))
  resolved_planner = (
    output.with_suffix(".planner.yaml")
    if output.suffix == ".jsonl"
    else output / "hope_planner_optimized_2.yaml"
  )
  resolved_planner.parent.mkdir(parents=True, exist_ok=True)
  resolved_planner.write_bytes(planner_config.read_bytes())
  if args.env == "mujoco":
    from deploy.g1_pingpong.feeder import RandomBallFeeder
    from deploy.g1_pingpong.simulation_runtime import run_deployment
    from deploy.hope_g1.feeder import MotionPlaneFeeder
    from deploy.hope_g1.random_feeder import RandomLaunchFeeder

    feed_mode = options["feed"].get("mode", "point")
    feeders = dict(
      point=RandomBallFeeder,
      motion_plane=MotionPlaneFeeder,
      random_launch=RandomLaunchFeeder,
    )
    if feed_mode not in feeders:
      parser.error("HOPE feed mode must be point, motion_plane or random_launch")

    result = run_deployment(
      deployment,
      args.real_config,
      args.mocap_config,
      controller_factory=HopeController,
      motion_index=args.motion_index,
      motion_select=args.motion_select,
      motion_allow=allow,
      simulation_config=resolved_rally,
      duration=args.duration,
      episodes=args.episodes,
      seed=args.seed,
      render=args.render,
      controls=args.sim_controls,
      startup=args.sim_startup,
      standing_only=args.standing_only,
      ball_source=args.ball_source or "sim",
      output=output,
      minimum_hit_rate=args.min_hit_rate,
      minimum_return_rate=args.min_return_rate,
      feeder_factory=feeders[feed_mode],
    )
    print(
      json.dumps(
        {
          k: result.get(k)
          for k in (
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
            "physics_validation",
          )
        },
        indent=2,
      )
    )
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
      controller_factory=HopeController,
      enable_motors=args.enable_motors,
      motion_index=args.motion_index,
      motion_select=args.motion_select,
      motion_allow=allow,
      duration=args.duration,
      output=output,
      simulation_config=resolved_rally,
    )


if __name__ == "__main__":
  main()

"""Run a selected G1 checkpoint directly in MuJoCo or on real hardware."""

import argparse
import signal
from dataclasses import replace
from pathlib import Path

import yaml

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments


def request_shutdown(signum, frame):
  raise KeyboardInterrupt("Deployment shutdown requested")


def main():
  signal.signal(signal.SIGTERM, request_shutdown)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env", choices=["mujoco", "real"], required=True)
  add_policy_arguments(parser)
  parser.add_argument(
    "--motion-index",
    type=int,
    default=19,
    help="Preparation/return home; also the swing ID with --motion-select fixed",
  )
  parser.add_argument(
    "--motion-select",
    choices=["fixed", "auto"],
    default="auto",
    help="Default: auto (chooses swings from --motion-allow); fixed uses --motion-index",
  )
  parser.add_argument(
    "--motion-allow", help="Candidate IDs, e.g. 5,13 or all; default from sim config"
  )
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--all-motions", action="store_true")
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--fixed-speed", type=float)
  parser.add_argument("--zero-residual", action="store_true")
  parser.add_argument("--ball-source", choices=["sim", "mocap"])
  parser.add_argument(
    "--sim-mode",
    choices=["episode", "full", "standing"],
    default="full",
    help="episode: isolated training evaluation; full/standing: shared real deployment loop",
  )
  parser.add_argument("--sim-controls", choices=["auto", "manual"], default="auto")
  parser.add_argument(
    "--sim-startup", choices=["suspended", "ground"], default="suspended"
  )
  parser.add_argument("--sim-stable-seconds", type=float, default=3.0)
  parser.add_argument(
    "--sim-policy-wait",
    type=float,
    default=0.0,
    help="Optional delay before the first simulated ball only; default zero",
  )
  parser.add_argument(
    "--sim-feed-interval",
    type=float,
    default=None,
    help="Seconds between independent simulated feeds (default 2)",
  )
  parser.add_argument("--sim-physics-dt", type=float, default=0.001)
  parser.add_argument(
    "--real-config", type=Path, default=Path("deploy/configs/real.yaml")
  )
  parser.add_argument(
    "--mocap-config", type=Path, default=Path("deploy/configs/mocap.yaml")
  )
  parser.add_argument("--duration", type=float, default=120.0)
  parser.add_argument(
    "--final-hold-seconds",
    type=float,
    default=5.0,
    help="Required stable hold after the last MuJoCo rally recovery",
  )
  parser.add_argument(
    "--rally-config",
    "--sim-config",
    dest="sim_config",
    type=Path,
    default=Path("deploy/configs/mujoco.yaml"),
    help="Shared MuJoCo/real receiving configuration",
  )
  parser.add_argument("--enable-motors", action="store_true")
  parser.add_argument(
    "--network-interface", help="Override the G1 DDS network interface"
  )
  parser.add_argument("--torch-threads", type=int, default=1)
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(parser)
  args = parser.parse_args()
  configure_viewer(args)
  args.ball_source = args.ball_source or ("mocap" if args.env == "real" else "sim")
  if args.env == "real" and args.ball_source != "mocap":
    parser.error("Real deployment requires measured mocap ball positions")
  if args.torch_threads < 1:
    parser.error("torch-threads must be positive")
  import torch

  torch.set_num_threads(args.torch_threads)
  if args.motion_index < 0 or args.episodes < 1 or args.duration <= 0:
    parser.error("Invalid motion index, episodes or duration")
  if args.env == "real":
    from deploy.backends.sdk import network_interfaces, require_sdk

    require_sdk()
    interface = (
      args.network_interface
      or yaml.safe_load(args.real_config.read_text())["network_interface"]
    )
    if interface not in network_interfaces():
      parser.error(
        f"Network interface {interface!r} does not exist; use --network-interface"
      )
  deployment = from_arguments(args)
  if args.motion_index >= len(deployment.motions):
    parser.error("motion-index exceeds the selected motion library")
  print(f"Policy: {deployment.checkpoint}; task: {deployment.settings['task']}")
  if args.env == "mujoco":
    if args.sim_mode in ("full", "standing"):
      from deploy.g1_pingpong.simulation_runtime import run_deployment

      if args.all_motions or args.fixed_speed is not None or args.zero_residual:
        parser.error(
          "Rally mode uses one home motion and optional automatic selection; no ablations"
        )
      allow = (
        (
          list(range(len(deployment.motions)))
          if args.motion_allow == "all"
          else [int(value) for value in args.motion_allow.split(",")]
        )
        if args.motion_allow
        else None
      )
      result = run_deployment(
        deployment,
        args.real_config,
        args.mocap_config,
        motion_index=args.motion_index,
        episodes=args.episodes,
        duration=args.duration,
        render=args.render,
        seed=args.seed,
        output=args.output,
        standing_only=args.sim_mode == "standing",
        simulation_config=args.sim_config,
        motion_select=args.motion_select,
        motion_allow=allow,
        ball_source=args.ball_source,
        final_hold_seconds=args.final_hold_seconds,
        controls=args.sim_controls,
        startup=args.sim_startup,
        stable_seconds=args.sim_stable_seconds,
        policy_wait=args.sim_policy_wait,
        physics_dt=args.sim_physics_dt,
        feed_interval=args.sim_feed_interval,
      )
      if not result["passed"]:
        raise SystemExit(1)
      return
    if args.ball_source != "sim":
      parser.error("External ball positions use --sim-mode full")
    from deploy.g1_pingpong.commands import policy_pd_settings
    from deploy.tools.evaluate import run_mujoco

    deployment = replace(
      deployment,
      settings=policy_pd_settings(
        deployment.settings, yaml.safe_load(args.real_config.read_text())
      ),
    )
    run_mujoco(
      deployment,
      motion_index=args.motion_index,
      episodes=args.episodes,
      all_motions=args.all_motions,
      render=args.render,
      seed=args.seed,
      output=args.output,
      fixed_speed=args.fixed_speed,
      zero_residual=args.zero_residual,
    )
  else:
    from deploy.backends.unitree import run_real

    if (
      args.sim_mode not in ("episode", "full")
      or args.fixed_speed is not None
      or args.zero_residual
      or args.all_motions
      or args.render
    ):
      parser.error("Ablations, batch evaluation and render are MuJoCo options")
    from deploy.tools.preflight import check_configuration

    cfg, _ = check_configuration(
      args.real_config, args.mocap_config, args.network_interface
    )
    run_real(
      deployment,
      cfg,
      args.mocap_config,
      enable_motors=args.enable_motors,
      motion_index=args.motion_index,
      duration=args.duration,
      output=args.output,
      simulation_config=args.sim_config,
      motion_select=args.motion_select,
      motion_allow=(
        list(range(len(deployment.motions)))
        if args.motion_allow == "all"
        else [int(value) for value in args.motion_allow.split(",")]
      )
      if args.motion_allow
      else None,
    )


if __name__ == "__main__":
  main()

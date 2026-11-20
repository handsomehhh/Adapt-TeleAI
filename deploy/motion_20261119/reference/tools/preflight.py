"""Validate deployment configuration and optional model artifacts without IO."""

import argparse
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

from deploy.motion_20261119.reference.backends.sdk import network_interfaces, sdk_status
from deploy.motion_20261119.reference.backends.unitree import SDK_JOINT_NAMES
from deploy.motion_20261119.reference.g1_pingpong.commands import policy_pd_settings
from deploy.motion_20261119.reference.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.motion_20261119.reference.g1_pingpong.core import rotation
from deploy.motion_20261119.reference.g1_pingpong.pd_gains import deployment_gains
from deploy.motion_20261119.reference.io.mocap import read_config


def check_configuration(real_config, mocap_config, network_interface=None):
  cfg = yaml.safe_load(Path(real_config).read_text())
  from deploy.motion_20261119.reference.g1_pingpong.hope_safety import real_stop_settings

  real_stop_settings(cfg)  # Validate profile before opening any robot IO.
  if network_interface is not None:
    cfg["network_interface"] = network_interface
  mocap, _ = read_config(mocap_config)
  for name in (
    "connection_timeout",
    "state_timeout",
    "command_timeout",
    "halt_kd",
    "transition_seconds",
    "max_control_overrun",
    "max_tilt_degrees",
  ):
    if not np.isfinite(cfg[name]) or cfg[name] <= 0:
      raise ValueError(f"real.{name} must be finite and positive")
  for name in ("max_tracking_error", "max_measured_velocity", "max_target_velocity"):
    if not np.isfinite(cfg["guard"][name]) or cfg["guard"][name] <= 0:
      raise ValueError(f"guard.{name} must be positive")
  for name in (
    "enforce_target_limits",
    "enforce_target_tracking",
    "enforce_target_slew",
  ):
    if name in cfg["guard"] and not isinstance(cfg["guard"][name], bool):
      raise ValueError(f"guard.{name} must be boolean")
  for name in ("measured_limit_margin", "max_measured_step"):
    if name in cfg["guard"] and (
      not np.isfinite(cfg["guard"][name]) or cfg["guard"][name] <= 0
    ):
      raise ValueError(f"guard.{name} must be positive")
  rotation(cfg["imu_to_pelvis_quaternion_wxyz"])
  policy_pd_settings(
    {"robot": {"kp": [1.0] * 29, "kd": [1.0] * 29, "joint_names": SDK_JOINT_NAMES}}, cfg
  )
  deployment_gains(
    SDK_JOINT_NAMES, profile="stable", scales=cfg.get("stable_policy", {}).get("pd", {})
  )
  poll = cfg.get("io_poll_seconds", 0.001)
  if not np.isfinite(poll) or not 0 < poll <= min(
    cfg["state_timeout"], cfg["command_timeout"]
  ):
    raise ValueError("io_poll_seconds must fit within the DDS watchdog leases")
  if not cfg["network_interface"] or cfg["expected_mode_machine"] != 5:
    raise ValueError("Configure a network interface and unlocked-waist G1 mode 5")
  return cfg, mocap


def report(
  real_config, mocap_config, deployment=None, require_real=False, network_interface=None
):
  cfg, mocap = check_configuration(real_config, mocap_config, network_interface)
  packages = {}
  for name in ("numpy", "torch", "mujoco", "PyYAML"):
    try:
      packages[name] = version(name)
    except PackageNotFoundError:
      packages[name] = None
  sdk = importlib.util.find_spec("unitree_sdk2py") is not None
  sdk_probe = sdk_status() if require_real else None
  interfaces = network_interfaces()
  ros = importlib.util.find_spec("rclpy") is not None
  valid = all(packages.values()) and (
    not require_real
    or (
      sdk_probe is not None
      and sdk_probe["available"]
      and cfg["network_interface"] in interfaces
      and (mocap["transport"] != "ros2" or ros)
    )
  )
  result = {
    "valid": bool(valid),
    "packages": packages,
    "sdk2_installed": sdk,
    "sdk2_import_check": sdk_probe,
    "network_interface": cfg["network_interface"],
    "available_interfaces": interfaces,
    "ros2_installed": ros,
    "mocap_transport": mocap["transport"],
    "mocap_source": mocap.get(
      "source",
      {
        "ball_topic": mocap["ball_topic"],
        "root_topic": mocap["root_topic"],
        "ball_type": "point_stamped",
        "frame": mocap["world_frame"],
      },
    ),
    "mocap_source_transform": mocap["transform"] if "source" in mocap else None,
    "dds_io_process": cfg.get("io_process", True),
    "robot_commands_sent": False,
  }
  if deployment is not None:
    result["policy"] = deployment.report()
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--real-config", type=Path, default=Path("deploy/configs/real.yaml")
  )
  parser.add_argument(
    "--mocap-config", type=Path, default=Path("deploy/configs/mocap.yaml")
  )
  add_policy_arguments(parser)
  parser.add_argument("--require-real", action="store_true")
  parser.add_argument("--network-interface")
  args = parser.parse_args()
  deployment = from_arguments(args) if args.checkpoint or args.config else None
  result = report(
    args.real_config,
    args.mocap_config,
    deployment,
    args.require_real,
    args.network_interface,
  )
  print(json.dumps(result, indent=2))
  if not result["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()

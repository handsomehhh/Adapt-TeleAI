"""Inspect physical A3 or G1 Stage-1 paddle tracking at each annotated strike.

Runs all supplied clips from frame zero at reference speed 1. Reports global
and anchor-aligned paddle errors, actual/reference states at the strike, and
whether a nearby physical-time sample reaches the intended contact point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_a3_pingpong import reset_all_clips, resolve_jit

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.adapt_tennis.mdp.commands import _resolve_motion_paths
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply


def racket_state(position, quaternion, lin_velocity, ang_velocity, offset):
  offset_w = quat_apply(quaternion, offset)
  normal_local = torch.zeros_like(offset)
  normal_local[:, 1] = 1
  return (
    position + offset_w,
    lin_velocity + torch.cross(ang_velocity, offset_w, dim=-1),
    quat_apply(quaternion, normal_local),
  )


@torch.inference_mode()
def diagnose(env, policy, files, seed):
  motion, obs = reset_all_clips(env, seed)
  n = env.num_envs
  wrist_names = [name for name in motion.cfg.body_names if "right_wrist_yaw" in name]
  if len(wrist_names) != 1:
    raise ValueError(f"Expected one right wrist in tracked bodies: {wrist_names}")
  wrist_name = wrist_names[0]
  wrist = motion.cfg.body_names.index(wrist_name)
  robot_wrist = motion.robot.body_names.index(wrist_name)
  with np.load(files[0], allow_pickle=False) as archive:
    mount_offset = np.asarray(archive["racket_mount_offset"], dtype=np.float32)
  offset = torch.tensor(mount_offset, device=env.device).expand(n, 3)
  states = {
    name: []
    for name in (
      "actual_position",
      "reference_position",
      "aligned_reference_position",
      "actual_velocity",
      "reference_velocity",
      "actual_normal",
      "reference_normal",
      "root_error",
      "alive",
    )
  }
  alive = torch.ones(n, dtype=torch.bool, device=env.device)
  times = []
  durations = []
  for file in files:
    with np.load(file, allow_pickle=False) as archive:
      fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
      durations.append((len(archive["joint_pos"]) - 1) / fps)
  max_time = max(durations)
  for index in range(round(max_time / env.step_dt) + 1):
    actual = racket_state(
      motion.robot.data.body_link_pos_w[:, robot_wrist],
      motion.robot.data.body_link_quat_w[:, robot_wrist],
      motion.robot.data.body_link_lin_vel_w[:, robot_wrist],
      motion.robot.data.body_link_ang_vel_w[:, robot_wrist],
      offset,
    )
    reference = racket_state(
      motion.body_pos_w[:, wrist],
      motion.body_quat_w[:, wrist],
      motion.body_lin_vel_w[:, wrist],
      motion.body_ang_vel_w[:, wrist],
      offset,
    )
    aligned = motion.body_pos_relative_w[:, wrist] + quat_apply(
      motion.body_quat_relative_w[:, wrist], offset
    )
    values = {
      "actual_position": actual[0] - env.scene.env_origins,
      "reference_position": reference[0] - env.scene.env_origins,
      "aligned_reference_position": aligned - env.scene.env_origins,
      "actual_velocity": actual[1],
      "reference_velocity": reference[1],
      "actual_normal": actual[2],
      "reference_normal": reference[2],
      "root_error": motion.robot_anchor_pos_w - motion.anchor_pos_w,
      "alive": alive,
    }
    for name, value in values.items():
      states[name].append(value.clone().cpu().numpy())
    times.append(index * env.step_dt)
    if index == round(max_time / env.step_dt):
      break
    obs, _, terminated, truncated, _ = env.step(policy(obs["actor"]))
    alive &= ~(terminated | truncated)
  arrays = {name: np.stack(value) for name, value in states.items()}
  times = np.asarray(times)
  rows = []
  for i, file in enumerate(files):
    with np.load(file) as archive:
      strike_time = float(archive["strike_time_s"])
    strike = int(np.argmin(abs(times - strike_time)))
    actual_p = arrays["actual_position"][strike, i]
    target_p = arrays["reference_position"][strike, i]
    delta = actual_p - target_p
    aligned_delta = actual_p - arrays["aligned_reference_position"][strike, i]
    valid_window = (abs(times - strike_time) <= 0.3) & arrays["alive"][:, i]
    window_ids = np.flatnonzero(valid_window)
    distances = np.linalg.norm(
      arrays["actual_position"][window_ids, i] - target_p, axis=-1
    )
    best = int(window_ids[distances.argmin()]) if len(window_ids) else strike
    # This nearest-trajectory estimate diagnoses a timing lag; it is not a
    # phase observation available to the deployed receiving actor.
    ref_window = np.flatnonzero(abs(times - strike_time) <= 0.3)
    phase_distances = np.linalg.norm(
      arrays["reference_position"][ref_window, i] - actual_p, axis=-1
    )
    nearest_phase = int(ref_window[phase_distances.argmin()])
    normal_dot = np.dot(
      arrays["actual_normal"][strike, i], arrays["reference_normal"][strike, i]
    )
    row = {
      "motion": file.name,
      "alive_at_strike": bool(arrays["alive"][strike, i]),
      "strike_time_s": strike_time,
      "sample_time_s": float(times[strike]),
      "position_error_m": float(np.linalg.norm(delta)),
      "position_error_xyz_m": delta.tolist(),
      "anchor_aligned_position_error_m": float(np.linalg.norm(aligned_delta)),
      "actual_position_w": actual_p.tolist(),
      "reference_position_w": target_p.tolist(),
      "actual_velocity_w": arrays["actual_velocity"][strike, i].tolist(),
      "reference_velocity_w": arrays["reference_velocity"][strike, i].tolist(),
      "velocity_error_m_s": float(
        np.linalg.norm(
          arrays["actual_velocity"][strike, i] - arrays["reference_velocity"][strike, i]
        )
      ),
      "actual_speed_m_s": float(np.linalg.norm(arrays["actual_velocity"][strike, i])),
      "reference_speed_m_s": float(
        np.linalg.norm(arrays["reference_velocity"][strike, i])
      ),
      "normal_error_deg": float(np.rad2deg(np.arccos(np.clip(normal_dot, -1, 1)))),
      "anchor_translation_error_xyz_m": arrays["root_error"][strike, i].tolist(),
      "nearest_target_distance_within_300ms_m": float(
        np.linalg.norm(arrays["actual_position"][best, i] - target_p)
      ),
      "nearest_target_physical_time_s": float(times[best]),
      "nearest_reference_phase_at_strike_s": float(times[nearest_phase]),
      "apparent_reference_phase_lag_s": float(strike_time - times[nearest_phase]),
    }
    rows.append(row)
  valid = [row for row in rows if row["alive_at_strike"]]
  summary = {"clips": n, "alive_at_strike": len(valid)}
  for name in (
    "position_error_m",
    "anchor_aligned_position_error_m",
    "velocity_error_m_s",
    "actual_speed_m_s",
    "reference_speed_m_s",
    "normal_error_deg",
    "nearest_target_distance_within_300ms_m",
    "apparent_reference_phase_lag_s",
  ):
    values = [row[name] for row in valid]
    summary[name] = (
      {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
      }
      if values
      else None
    )
  summary["within_10cm_at_strike"] = sum(
    row["position_error_m"] <= 0.10 for row in valid
  )
  summary["within_10cm_at_any_time_300ms"] = sum(
    row["nearest_target_distance_within_300ms_m"] <= 0.10 for row in valid
  )
  return {"summary": summary, "clips": rows}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-PingPong-Tracking-A3-Stage1-RandomDt")
  parser.add_argument("--motion-directory", type=Path)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  cfg = load_env_cfg(args.task, play=True)
  if args.motion_directory is None:
    files = [
      Path(path).resolve() for path in _resolve_motion_paths(cfg.commands["motion"])
    ]
  else:
    files = sorted(args.motion_directory.resolve().glob("*.npz"))
  if not files:
    parser.error("No motion archives found.")
  cfg.scene.num_envs = len(files)
  cfg.seed = args.seed
  cfg.events = {}
  cfg.commands["motion"].motion_files = tuple(str(path) for path in files)
  cfg.commands["motion"].random_dt_training_enabled = False
  cfg.commands["motion"].debug_vis = False
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  report = {
    "seed": args.seed,
    "protocol": "All clips at frame zero, reference speed 1, strike time from NPZ; physical and reference timeline both start at zero.",
    "checkpoints": [],
  }
  try:
    for checkpoint in args.checkpoint:
      jit = resolve_jit(checkpoint)
      policy = torch.jit.load(str(jit), map_location=args.device).eval()
      result = diagnose(env, policy, files, args.seed)
      result["checkpoint"] = str(checkpoint)
      report["checkpoints"].append(result)
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2) + "\n")
      print(str(checkpoint), json.dumps(result["summary"]))
  finally:
    env.close()


if __name__ == "__main__":
  main()

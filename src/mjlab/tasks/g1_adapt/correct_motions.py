"""Create physics-corrected MVAE data using a newly trained stage-1 tracker."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro

from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_stage1.env_cfg import g1_stage1_env_cfg
from mjlab.tasks.g1_stage1.mdp.tracking import (
  Stage1MotionCommand,
  Stage1MotionCommandCfg,
)

from .mvae import file_hash


@dataclass
class CorrectMotions:
  tracker: str
  motion_directory: str = "dataset/g1_stage1/motions"
  output_directory: str = "dataset/g1_adapt_corrected/motions"
  device: str = "cuda:0"
  max_clips: int = 0
  max_joint_rms: float = 0.5
  max_wrist_error: float = 0.3


def correct(cfg):
  paths = sorted(Path(cfg.motion_directory).resolve().rglob("*.npz"))
  if cfg.max_clips:
    paths = paths[: cfg.max_clips]
  if not paths:
    raise FileNotFoundError(cfg.motion_directory)
  if Path(cfg.output_directory).resolve() == Path(cfg.motion_directory).resolve():
    raise ValueError("Keep raw and corrected data in separate directories")
  if list(Path(cfg.output_directory).glob("*.npz")):
    raise ValueError("Use an empty output directory to prevent stale corrected clips")
  tracker = torch.jit.load(cfg.tracker, map_location=cfg.device).eval()
  config = g1_stage1_env_cfg(play=True)
  config.scene.num_envs = len(paths)
  motion_cfg = cast(Stage1MotionCommandCfg, config.commands["motion"])
  motion_cfg.motion_files = tuple(map(str, paths))
  motion_cfg.motion_directory = ""
  motion_cfg.debug_vis = False
  config.terminations = {}
  env = ManagerBasedRlEnv(config, device=cfg.device)
  output = Path(cfg.output_directory)
  output.mkdir(parents=True, exist_ok=True)
  try:
    env.reset()
    command = env.command_manager.get_term("motion")
    assert isinstance(command, Stage1MotionCommand)
    ids = torch.arange(len(paths), device=cfg.device)
    command.motion_ids[:] = ids
    command.reset_to_frame(ids, 0)
    env.sim.forward()
    lengths = command.motion.lengths.cpu().tolist()
    fields = {
      key: []
      for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      )
    }
    failed = torch.zeros(len(paths), dtype=torch.bool, device=cfg.device)
    max_joint_rms = torch.zeros(len(paths), device=cfg.device)
    max_wrist_error = torch.zeros_like(max_joint_rms)
    robot = env.scene["robot"]
    for step in range(max(lengths)):
      data = robot.data
      values = {
        "joint_pos": data.joint_pos,
        "joint_vel": data.joint_vel,
        "body_pos_w": data.body_link_pos_w - env.scene.env_origins[:, None],
        "body_quat_w": data.body_link_quat_w,
        "body_lin_vel_w": data.body_link_lin_vel_w,
        "body_ang_vel_w": data.body_link_ang_vel_w,
      }
      for key, value in values.items():
        fields[key].append(value.detach().cpu().numpy().copy())
      valid_time = step < command.motion.lengths
      height_error = (data.root_link_pos_w[:, 2] - command.body_pos_w[:, 0, 2]).abs()
      joint_rms = (
        (data.joint_pos[:, command._motion_to_robot_joint_ids] - command.joint_pos)
        .square()
        .mean(-1)
        .sqrt()
      )
      wrist = motion_cfg.body_names.index("right_wrist_yaw_link")
      actual_wrist = robot.body_names.index("right_wrist_yaw_link")
      wrist_error = (
        data.body_link_pos_w[:, actual_wrist] - command.body_pos_w[:, wrist]
      ).norm(dim=-1)
      max_joint_rms = torch.maximum(
        max_joint_rms, torch.where(valid_time, joint_rms, 0)
      )
      max_wrist_error = torch.maximum(
        max_wrist_error, torch.where(valid_time, wrist_error, 0)
      )
      failed |= valid_time & (
        (height_error > 0.25)
        | (data.projected_gravity_b[:, 2] > -0.65)
        | (joint_rms > cfg.max_joint_rms)
        | (wrist_error > cfg.max_wrist_error)
      )
      with torch.no_grad():
        actions = tracker(env.observation_manager.compute_group("actor"))
      if not torch.isfinite(actions).all():
        raise RuntimeError("Tracker produced non-finite correction actions")
      env.step(actions)
    records = []
    for index, path in enumerate(paths):
      record = {
        "source": str(path),
        "source_sha256": file_hash(path),
        "accepted": not bool(failed[index]),
        "max_joint_rms": float(max_joint_rms[index]),
        "max_wrist_error": float(max_wrist_error[index]),
      }
      if record["accepted"]:
        with np.load(path, allow_pickle=True) as archive:
          metadata = {
            key: archive[key]
            for key in ("hit_type", "strike_frame", "strike_time_s")
            if key in archive
          }
        arrays = {
          key: np.stack(values)[: lengths[index], index]
          for key, values in fields.items()
        }
        destination = output / path.name
        np.savez_compressed(
          destination,
          allow_pickle=False,
          fps=np.array(50.0),
          joint_names=np.asarray(robot.joint_names),
          body_names=np.asarray(robot.body_names),
          adapt_physics_corrected=np.array(True),
          **arrays,
          **metadata,
        )
        record.update(file=destination.name, sha256=file_hash(destination))
      records.append(record)
    manifest = {
      "tracker_sha256": file_hash(cfg.tracker),
      "clips": records,
      "accepted": sum(bool(r["accepted"]) for r in records),
      "total": len(records),
    }
    (output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "clips"}), flush=True)
    if manifest["accepted"] < 2:
      raise RuntimeError(
        "Fewer than two complete stable clips. Train/validate Stage 1 further before training MVAE."
      )
    return manifest
  finally:
    env.close()


def main():
  correct(tyro.cli(CorrectMotions, config=TYRO_FLAGS))


if __name__ == "__main__":
  main()

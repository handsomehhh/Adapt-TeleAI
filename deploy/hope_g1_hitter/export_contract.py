"""Export the G1 hitter training contract to JSON, from the training repository.

Run with the PingPong mjlab interpreter, which can import ``hope_mjlab``:

  ~/TeleHuman/PingPong/hope_training/mjlab_training/.venv/bin/python \
    deploy/hope_g1_hitter/export_contract.py deploy/hope_g1_hitter/artifacts/<run>

The deploy side then reads only this JSON, the checkpoint and the NPZ clips.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("artifact_dir", type=Path)
  args = parser.parse_args()
  from hope_mjlab import g1_robot

  arrays = g1_robot.nominal_arrays()
  checkpoints = list(args.artifact_dir.glob("model_*.pt"))
  if len(checkpoints) != 1:
    raise ValueError("Expected exactly one model_*.pt in artifact_dir")
  checkpoint = checkpoints[0]
  raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
  from hope_mjlab.training_play import TrainingPlayProfile

  profile = TrainingPlayProfile(raw)
  command = profile.command
  rows = g1_robot.reference_metadata()
  motions = [Path(p) for p in g1_robot.motion_files()]
  for source in motions:
    copied = args.artifact_dir / "motions" / source.name
    if sha256(source) != sha256(copied):
      raise ValueError(f"Copied clip differs from the training reference: {copied}")
  contract = dict(
    actor_obs_contract="g1_hitter_pure_v1",
    action_contract="default_pose_plus_scaled_action_soft_clamp_v1",
    checkpoint_sha256=sha256(checkpoint),
    iteration=int(raw["iter"]),
    task_identity=raw["infos"]["hope_recipe"].get("task_identity", {}),
    control_dt=0.02,
    joint_names=list(g1_robot.POLICY_JOINT_NAMES),
    default_q=arrays["default_q"].tolist(),
    action_scale=profile.action_scale.tolist(),
    kp=arrays["kp"].tolist(),
    kd=arrays["kd"].tolist(),
    effort=arrays["effort"].tolist(),
    hard_lower=arrays["hard_lower"].tolist(),
    hard_upper=arrays["hard_upper"].tolist(),
    # mjlab soft_joint_pos_limit_factor=0.9 around the hard-range centre.
    soft_limit_factor=0.9,
    ready_height=g1_robot.ready_height(),
    # Training world: robot station at origin facing +x, floor at z=0.
    racket_pos_range_per_clip=profile.core_positions.tolist(),
    racket_vel_range_per_clip=profile.final_velocities.tolist(),
    position_curriculum_progress=profile.position_progress,
    windup=profile.windup.tolist(),
    follow=profile.follow.tolist(),
    racket_mount_offset=list(command["mount_offset"]),
    racket_mount_quat=list(command["mount_quat"]),
    robot_xml="robot/g1.xml",
    robot_xml_sha256=sha256(args.artifact_dir / "robot/g1.xml"),
    base_target_y_range=list(command["base_target_y_range"]),
    clips=[
      dict(
        side=side,
        file=path.name,
        sha256=sha256(path),
        frames=row["frames"],
        strike_frame=int(round(profile.windup[i] / 0.02)),
        strike_position=row["position"].tolist(),
        strike_velocity=row["velocity"].tolist(),
      )
      for i, (side, path, row) in enumerate(
        zip(("forehand", "backhand"), motions, rows, strict=True)
      )
    ],
  )
  (args.artifact_dir / "contract.json").write_text(
    json.dumps(contract, indent=2) + "\n"
  )
  files = sorted(
    p for p in args.artifact_dir.rglob("*") if p.is_file() and p.name != "manifest.json"
  )
  manifest = [
    dict(
      path=str(p.relative_to(args.artifact_dir)),
      bytes=p.stat().st_size,
      sha256=sha256(p),
    )
    for p in files
  ]
  (args.artifact_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
  )
  print(f"Wrote contract.json and manifest.json ({len(manifest)} files)")


if __name__ == "__main__":
  main()

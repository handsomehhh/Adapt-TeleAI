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

import numpy as np


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("artifact_dir", type=Path)
  args = parser.parse_args()
  from hope_mjlab import g1_robot

  arrays = g1_robot.nominal_arrays()
  command = g1_robot.RECIPE["commands"]["racket_target"]
  rows = g1_robot.reference_metadata()
  motions = [Path(p) for p in g1_robot.motion_files()]
  for source in motions:
    copied = args.artifact_dir / "motions" / source.name
    if sha256(source) != sha256(copied):
      raise ValueError(f"Copied clip differs from the training reference: {copied}")
  contract = dict(
    actor_obs_contract="g1_hitter_pure_v1",
    action_contract="default_pose_plus_scaled_action_soft_clamp_v1",
    control_dt=0.02,
    joint_names=list(g1_robot.POLICY_JOINT_NAMES),
    default_q=arrays["default_q"].tolist(),
    action_scale=arrays["action_scale"].tolist(),
    kp=arrays["kp"].tolist(),
    kd=arrays["kd"].tolist(),
    effort=arrays["effort"].tolist(),
    hard_lower=arrays["hard_lower"].tolist(),
    hard_upper=arrays["hard_upper"].tolist(),
    # mjlab soft_joint_pos_limit_factor=0.9 around the hard-range centre.
    soft_limit_factor=0.9,
    ready_height=g1_robot.ready_height(),
    # Training world: robot station at origin facing +x, floor at z=0.
    racket_pos_range_per_clip=np.asarray(command["racket_pos_range_per_clip"]).tolist(),
    racket_vel_range_per_clip=np.asarray(command["racket_vel_range_per_clip"]).tolist(),
    base_target_y_range=list(command["base_target_y_range"]),
    clips=[
      dict(
        side=side,
        file=path.name,
        sha256=sha256(path),
        frames=row["frames"],
        strike_frame=row["strike"],
        strike_position=row["position"].tolist(),
        strike_velocity=row["velocity"].tolist(),
      )
      for side, path, row in zip(("forehand", "backhand"), motions, rows, strict=True)
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

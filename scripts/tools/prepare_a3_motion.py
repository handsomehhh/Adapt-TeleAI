"""Validate and canonicalize the supplied named A3 ping-pong motion archives.

The input already uses the AdaPT ``MotionLoader`` schema (world-frame velocities,
wxyz quaternions). Keep raw files untouched; rotate every world-space field by
the inverse initial pelvis yaw, then translate initial pelvis XY to zero. Ground
height, joint coordinates, clip duration, and velocity magnitudes are unchanged.

Example (run from the repository root)::

  uv run --no-sync python scripts/tools/prepare_a3_motion.py \
    --source dataset/a3_pingpong/raw --output dataset/a3_pingpong/motions

The nominal strike is frame 43 in the HOPE peak-clipped 94-frame dataset. It is
not always the global maximum of the mounted paddle speed, so both are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT_BODY = "pelvis_link"
WRIST_BODY = "right_wrist_yaw_Link"
RACKET_OFFSET = np.array([0.210211399202899, 0.0320784994676765, 0.0320358706296689])
RACKET_NORMAL = np.array([0.0, 1.0, 0.0])
DEFAULT_SOURCE = Path("dataset/a3_pingpong/raw")
DEFAULT_OUTPUT = Path("dataset/a3_pingpong/motions")


def _names(data: dict[str, np.ndarray], field: str) -> list[str]:
  names = [str(item) for item in data[field].tolist()]
  if len(names) != len(set(names)):
    raise ValueError(f"Duplicate entries in {field}.")
  return names


def validate_motion(
  data: dict[str, np.ndarray],
  *,
  root_body: str = ROOT_BODY,
  wrist_body: str = WRIST_BODY,
) -> None:
  """Reject malformed input before it can silently poison PPO observations."""
  required = (
    "fps",
    "joint_pos",
    "joint_vel",
    "joint_names",
    "body_names",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  missing = set(required) - data.keys()
  if missing:
    raise ValueError(f"Missing required motion fields: {sorted(missing)}")
  if data["fps"].size != 1:
    raise ValueError("fps must be scalar or a length-one array.")
  fps = float(data["fps"].reshape(-1)[0])
  if not np.isfinite(fps) or fps <= 0:
    raise ValueError("fps must be finite and positive.")
  joints = _names(data, "joint_names")
  bodies = _names(data, "body_names")
  if root_body not in bodies or wrist_body not in bodies:
    raise ValueError(f"Motion must contain {root_body} and {wrist_body}.")
  frames = data["joint_pos"].shape[0]
  if frames < 3:
    raise ValueError("Motion must contain at least three frames.")
  expected = {
    "joint_pos": (frames, len(joints)),
    "joint_vel": (frames, len(joints)),
    "body_pos_w": (frames, len(bodies), 3),
    "body_quat_w": (frames, len(bodies), 4),
    "body_lin_vel_w": (frames, len(bodies), 3),
    "body_ang_vel_w": (frames, len(bodies), 3),
  }
  for key, shape in expected.items():
    if data[key].shape != shape:
      raise ValueError(f"{key}: expected {shape}, got {data[key].shape}.")
    if not np.isfinite(data[key]).all():
      raise ValueError(f"{key} contains NaN or infinite values.")
  norms = np.linalg.norm(data["body_quat_w"], axis=-1)
  if np.max(np.abs(norms - 1.0)) > 1e-3:
    raise ValueError("body_quat_w must contain normalized wxyz quaternions.")


def canonicalize_motion(
  data: dict[str, np.ndarray],
  *,
  strike_frame: int = 43,
  root_body: str = ROOT_BODY,
  wrist_body: str = WRIST_BODY,
  racket_offset: np.ndarray = RACKET_OFFSET,
  racket_normal: np.ndarray = RACKET_NORMAL,
  feet_bodies: tuple[str, ...] = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
  ),
) -> tuple[dict[str, np.ndarray], dict]:
  """Return independent arrays with +X initial heading and zero initial root XY."""
  validate_motion(data, root_body=root_body, wrist_body=wrist_body)
  result = {key: np.array(value, copy=True) for key, value in data.items()}
  bodies = _names(data, "body_names")
  root_idx, wrist_idx = bodies.index(root_body), bodies.index(wrist_body)
  frames = data["joint_pos"].shape[0]
  if not 0 <= strike_frame < frames:
    raise ValueError(f"strike_frame {strike_frame} outside clip of {frames} frames.")
  fps = float(data["fps"].reshape(-1)[0])
  root_pos = data["body_pos_w"][0, root_idx].astype(np.float64)
  root_rot = Rotation.from_quat(data["body_quat_w"][0, root_idx], scalar_first=True)
  initial_euler = root_rot.as_euler("xyz")
  yaw = float(initial_euler[2])
  yaw_correction = Rotation.from_euler("z", -yaw)
  # Translate before rotating about the world origin. Keep z fixed.
  origin = np.array([root_pos[0], root_pos[1], 0.0])
  pos = data["body_pos_w"].astype(np.float64) - origin
  result["body_pos_w"] = (
    yaw_correction.apply(pos.reshape(-1, 3)).reshape(pos.shape).astype(np.float32)
  )
  for key in ("body_lin_vel_w", "body_ang_vel_w"):
    result[key] = (
      yaw_correction.apply(data[key].reshape(-1, 3))
      .reshape(data[key].shape)
      .astype(np.float32)
    )
  if "root_link_lin_vel_w" in data:
    result["root_link_lin_vel_w"] = yaw_correction.apply(
      data["root_link_lin_vel_w"]
    ).astype(np.float32)
  rotations = yaw_correction * Rotation.from_quat(
    data["body_quat_w"].reshape(-1, 4), scalar_first=True
  )
  result["body_quat_w"] = (
    rotations.as_quat(scalar_first=True)
    .reshape(data["body_quat_w"].shape)
    .astype(np.float32)
  )
  wrist_rot = Rotation.from_quat(result["body_quat_w"][:, wrist_idx], scalar_first=True)
  offset_w = wrist_rot.apply(racket_offset)
  result["racket_pos_w"] = (result["body_pos_w"][:, wrist_idx] + offset_w).astype(
    np.float32
  )
  # Stored linear velocities are body-origin velocities, not COM velocities.
  result["racket_lin_vel_w"] = (
    result["body_lin_vel_w"][:, wrist_idx]
    + np.cross(result["body_ang_vel_w"][:, wrist_idx], offset_w)
  ).astype(np.float32)
  result["racket_normal_w"] = wrist_rot.apply(racket_normal).astype(np.float32)
  speed = np.linalg.norm(result["racket_lin_vel_w"], axis=-1)
  result.update(
    strike_frame=np.asarray(strike_frame, dtype=np.int64),
    strike_time_s=np.asarray(strike_frame / fps, dtype=np.float64),
    racket_speed_peak_frame=np.asarray(int(speed.argmax()), dtype=np.int64),
    racket_mount_offset=np.asarray(racket_offset).copy(),
    racket_normal_axis=np.asarray(1, dtype=np.int64),
    canonical_initial_yaw_correction_rad=np.asarray(-yaw, dtype=np.float64),
    canonical_source_origin_xy=origin[:2],
    canonical_coordinate_convention=np.asarray(
      "wxyz; world velocities; initial pelvis XY=0 and yaw=0; original Z"
    ),
  )
  summary = {
    "frames": frames,
    "fps": fps,
    "duration_s": (frames - 1) / fps,
    "strike_frame": strike_frame,
    "strike_time_s": strike_frame / fps,
    "racket_speed_peak_frame": int(speed.argmax()),
    "racket_speed_peak_m_s": float(speed.max()),
    "racket_speed_at_strike_m_s": float(speed[strike_frame]),
    "racket_pos_at_strike_m": result["racket_pos_w"][strike_frame].tolist(),
    "racket_normal_at_strike": result["racket_normal_w"][strike_frame].tolist(),
    "source_initial_root_xyz_m": root_pos.tolist(),
    "source_initial_root_rpy_deg": np.rad2deg(initial_euler).tolist(),
    "initial_root_height_m": float(root_pos[2]),
    "minimum_root_height_m": float(result["body_pos_w"][:, root_idx, 2].min()),
    "max_joint_speed_rad_s": float(np.abs(result["joint_vel"]).max()),
  }
  feet = [bodies.index(name) for name in feet_bodies if name in bodies]
  if feet:
    summary["initial_ankle_body_heights_m"] = result["body_pos_w"][0, feet, 2].tolist()
  return result, summary


def validate_fk(motions: list[Path], xml: Path, *, root_body: str = ROOT_BODY) -> dict:
  """Check named reference poses against first, strike and last model frames."""
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
  sim = mujoco.MjData(model)
  position_error = non_ankle_error = quat_error = 0.0
  checked = 0
  for path in motions:
    with np.load(path, allow_pickle=False) as archive:
      bodies = list(archive["body_names"])
      joints = list(archive["joint_names"])
      root = bodies.index(root_body)
      body_ids = [model.body(str(name)).id for name in bodies]
      qpos_ids = [int(model.jnt_qposadr[model.joint(str(name)).id]) for name in joints]
      non_ankle = [i for i, name in enumerate(bodies) if "ankle" not in name]
      frames = sorted({0, int(archive["strike_frame"]), len(archive["joint_pos"]) - 1})
      for frame in frames:
        sim.qpos[:] = model.qpos0
        sim.qpos[:3] = archive["body_pos_w"][frame, root]
        sim.qpos[3:7] = archive["body_quat_w"][frame, root]
        sim.qpos[qpos_ids] = archive["joint_pos"][frame]
        mujoco.mj_forward(model, sim)
        distances = np.linalg.norm(
          sim.xpos[body_ids] - archive["body_pos_w"][frame], axis=-1
        )
        position_error = max(position_error, float(distances.max()))
        non_ankle_error = max(non_ankle_error, float(distances[non_ankle].max()))
        qa, qb = sim.xquat[body_ids], archive["body_quat_w"][frame]
        quat_error = max(
          quat_error,
          float(
            np.minimum(
              np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1)
            ).max()
          ),
        )
        checked += 1
  # The archived Isaac motions omit the URDF's ±1.5 mm ankle offset.
  if position_error > 0.002 or non_ankle_error > 1e-4 or quat_error > 1e-4:
    raise ValueError(
      f"Model/motion FK mismatch: position={position_error}, non-ankle={non_ankle_error}, quat={quat_error}."
    )
  return {
    "xml": str(xml),
    "mujoco_version": mujoco.__version__,
    "frames_checked": checked,
    "max_body_position_error_m": position_error,
    "max_non_ankle_position_error_m": non_ankle_error,
    "max_body_quaternion_l2_error": quat_error,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--strike-frame", type=int, default=43)
  parser.add_argument(
    "--xml", type=Path, help="Optional robot MJCF for named FK validation."
  )
  args = parser.parse_args()
  if args.source.resolve() == args.output.resolve():
    parser.error("Source and output must differ to preserve raw archives.")
  sources = sorted(args.source.glob("*.npz"))
  if not sources:
    parser.error(f"No motion archives in {args.source}.")
  args.output.mkdir(parents=True, exist_ok=True)
  entries = []
  outputs = []
  for source in sources:
    with np.load(source, allow_pickle=False) as archive:
      result, summary = canonicalize_motion(
        dict(archive), strike_frame=args.strike_frame
      )
    target = args.output / source.name
    np.savez_compressed(target, **result)
    summary.update(
      file=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    entries.append(summary)
    outputs.append(target)
  manifest = {
    "format_version": 1,
    "source_directory": str(args.source),
    "motion_directory": str(args.output),
    "num_clips": len(entries),
    "total_frames": sum(entry["frames"] for entry in entries),
    "total_duration_s": sum(entry["duration_s"] for entry in entries),
    "quaternion_order": "wxyz",
    "velocity_frame": "world",
    "canonical_frame": "initial pelvis XY=(0,0), initial pelvis yaw=0, original ground Z",
    "strike_annotation": "HOPE nominal strike frame; global racket speed peak recorded separately",
    "clips": entries,
  }
  if args.xml:
    manifest["fk_validation"] = validate_fk(outputs, args.xml)
  destination = args.output.parent / "manifest.json"
  destination.write_text(json.dumps(manifest, indent=2) + "\n")
  print(
    f"Prepared {len(entries)} clips / {manifest['total_frames']} frames / {manifest['total_duration_s']:.2f} s in {args.output}"
  )
  print(f"Manifest: {destination}")
  if args.xml:
    print(json.dumps(manifest["fk_validation"], indent=2))


if __name__ == "__main__":
  main()

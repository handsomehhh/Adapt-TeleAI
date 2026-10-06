"""Canonicalize the authenticated 29-DoF G1 ping-pong reference clips.

The remote HOPE export faces approximately world -Y.  Training and the analytic
table use +X as the outgoing-ball direction, so this script rotates every world
pose and velocity by the inverse initial pelvis yaw and moves initial pelvis XY
to zero.  The original archives remain untouched in ``raw/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from prepare_a3_motion import canonicalize_motion, validate_fk
from scipy.spatial.transform import Rotation

ROOT_BODY = "pelvis"
WRIST_BODY = "right_wrist_yaw_link"
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
RACKET_OFFSET = np.array([0.215, 0.003, 0.0])
RACKET_NORMAL = np.array([0.0, 1.0, 0.0])
DEFAULT_SOURCE = Path("dataset/g1_pingpong/raw")
DEFAULT_OUTPUT = Path("dataset/g1_pingpong/motions")
DEFAULT_SOURCE_MANIFEST = Path("dataset/g1_pingpong/source_manifest.json")
DEFAULT_XML = Path("src/mjlab/asset_zoo/robots/unitree_g1_pingpong/g1.xml")
EXPECTED_CLIPS = 43
SUPPORTED_RETARGET_METHODS = frozenset(("bvh_mink_ik", "tele_gmr_football"))
COM_VELOCITY_CONVENTION = (
  "body COM world linear and angular velocity from G1 mj_jacBodyCom "
  "@ generalized velocity"
)


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sources(source: Path, manifest_path: Path) -> tuple[dict, list[Path]]:
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("format_version") != 2:
    raise ValueError("Expected a version-2 G1 retarget manifest.")
  if manifest.get("robot") != "G1":
    raise ValueError("Motion manifest is not for G1.")
  if manifest.get("retarget_method") not in SUPPORTED_RETARGET_METHODS:
    raise ValueError(
      "Unsupported G1 retarget method: "
      f"{manifest.get('retarget_method')!r}; expected one of "
      f"{sorted(SUPPORTED_RETARGET_METHODS)}."
    )
  if manifest.get("num_motions") != EXPECTED_CLIPS:
    raise ValueError(f"Expected exactly {EXPECTED_CLIPS} source clips.")
  acceptance = manifest.get("independent_acceptance", {})
  if (
    not acceptance.get("all_passed") or acceptance.get("passed_clips") != EXPECTED_CLIPS
  ):
    raise ValueError("The source manifest does not record 43 accepted clips.")
  records = {Path(record["path"]).name: record for record in manifest["clips"]}
  files = sorted(source.glob("*.npz"))
  if len(records) != EXPECTED_CLIPS or len(files) != EXPECTED_CLIPS:
    raise ValueError("The source manifest must contain 43 unique clips.")
  if set(records) != {path.name for path in files}:
    raise ValueError("Raw G1 clip set differs from the recorded 43-file manifest.")
  recorded_files = manifest.get("motion_files")
  if recorded_files is not None and {Path(name).name for name in recorded_files} != set(
    records
  ):
    raise ValueError("motion_files differs from the source clip records.")
  for path in files:
    if sha256(path) != records[path.name]["sha256"]:
      raise ValueError(f"Raw G1 checksum mismatch: {path}")
  hit_type_hash = manifest.get("hit_type_sha256")
  if hit_type_hash is not None:
    hit_type_file = manifest_path.parent / "hit_type.json"
    if not hit_type_file.is_file() or sha256(hit_type_file) != hit_type_hash:
      raise ValueError("G1 hit-type annotation checksum mismatch.")
  return manifest, files


def _source_velocity_convention(manifest: dict) -> str:
  """Return a verified description of the source body velocity convention."""
  declared = manifest.get("body_velocity_convention")
  if declared is not None:
    if not str(declared).startswith("body COM world linear and angular velocity"):
      raise ValueError(f"Unsupported body velocity convention: {declared!r}")
    return str(declared)

  # The Tele-GMR/Football export records the convention in every clip's
  # independent saved-FK validation rather than as a top-level string.
  if manifest.get("retarget_method") == "tele_gmr_football":
    for record in manifest["clips"]:
      validation = record.get("saved_fk_validation", {})
      if "max_body_com_lin_vel_error_m_s" not in validation:
        raise ValueError(
          f"{record.get('path')} does not record COM-velocity validation."
        )
    return COM_VELOCITY_CONVENTION
  raise ValueError("Source body velocity convention is not recorded.")


def _validate_source_archive(
  data: dict[str, np.ndarray],
  model: mujoco.MjModel,
  manifest: dict,
  record: dict,
  source_file: Path,
) -> dict[str, float]:
  """Validate schema, FK, and COM velocities before canonicalization."""
  required = {
    "body_ang_vel_w",
    "body_lin_vel_w",
    "body_names",
    "body_pos_w",
    "body_quat_w",
    "fps",
    "generalized_velocity",
    "joint_names",
    "joint_pos",
    "joint_vel",
    "root_link_lin_vel_w",
  }
  missing = required.difference(data)
  if missing:
    raise ValueError(f"{source_file} is missing fields: {sorted(missing)}")

  joint_names = tuple(str(name) for name in data["joint_names"].tolist())
  body_names = tuple(str(name) for name in data["body_names"].tolist())
  if joint_names != tuple(manifest["joint_names"]):
    raise ValueError(f"G1 joint schema mismatch: {source_file}")
  if body_names != tuple(manifest["body_names"]):
    raise ValueError(f"G1 body schema mismatch: {source_file}")

  expected_frames = int(record.get("output_frames", manifest["frames_per_motion"]))
  expected_fps = int(record.get("output_fps", manifest["fps"]))
  if data["joint_pos"].shape != (expected_frames, len(joint_names)):
    raise ValueError(f"Unexpected joint array shape: {source_file}")
  if data["body_pos_w"].shape != (expected_frames, len(body_names), 3):
    raise ValueError(f"Unexpected body array shape: {source_file}")
  if int(np.asarray(data["fps"]).reshape(-1)[0]) != expected_fps:
    raise ValueError(f"Unexpected source FPS: {source_file}")
  for key, value in data.items():
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
      raise ValueError(f"Non-finite {key} in {source_file}")

  joint_qpos_addresses = [
    int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names
  ]
  body_ids = [model.body(name).id for name in body_names]
  root_index = body_names.index(ROOT_BODY)
  mj_data = mujoco.MjData(model)
  max_position_error = 0.0
  max_com_velocity_error = 0.0
  max_root_link_velocity_error = 0.0
  strike_frame = int(record["strike_frame"])
  for frame in (0, strike_frame, expected_frames - 1):
    mj_data.qpos[:] = model.qpos0
    mj_data.qpos[:3] = data["body_pos_w"][frame, root_index]
    mj_data.qpos[3:7] = data["body_quat_w"][frame, root_index]
    mj_data.qpos[joint_qpos_addresses] = data["joint_pos"][frame]
    mj_data.qvel[:] = data["generalized_velocity"][frame]
    mujoco.mj_forward(model, mj_data)
    max_position_error = max(
      max_position_error,
      float(
        np.linalg.norm(
          mj_data.xpos[body_ids] - data["body_pos_w"][frame], axis=-1
        ).max()
      ),
    )
    for body_index, body_id in enumerate(body_ids):
      jacobian_position = np.zeros((3, model.nv))
      jacobian_rotation = np.zeros((3, model.nv))
      mujoco.mj_jacBodyCom(
        model,
        mj_data,
        jacobian_position,
        jacobian_rotation,
        body_id,
      )
      expected_velocity = jacobian_position @ mj_data.qvel
      max_com_velocity_error = max(
        max_com_velocity_error,
        float(
          np.linalg.norm(expected_velocity - data["body_lin_vel_w"][frame, body_index])
        ),
      )
    root_jacobian = np.zeros((3, model.nv))
    root_rotation_jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacBody(
      model,
      mj_data,
      root_jacobian,
      root_rotation_jacobian,
      body_ids[root_index],
    )
    max_root_link_velocity_error = max(
      max_root_link_velocity_error,
      float(
        np.linalg.norm(
          root_jacobian @ mj_data.qvel - data["root_link_lin_vel_w"][frame]
        )
      ),
    )
  if max_position_error > 2.0e-6:
    raise ValueError(f"Source FK mismatch in {source_file}: {max_position_error:.3e} m")
  if max_com_velocity_error > 3.0e-6:
    raise ValueError(
      f"Source COM velocity mismatch in {source_file}: {max_com_velocity_error:.3e} m/s"
    )
  if max_root_link_velocity_error > 3.0e-6:
    raise ValueError(
      f"Source root-link velocity mismatch in {source_file}: "
      f"{max_root_link_velocity_error:.3e} m/s"
    )
  return {
    "max_fk_position_error_m": max_position_error,
    "max_com_velocity_error_m_s": max_com_velocity_error,
    "max_root_link_velocity_error_m_s": max_root_link_velocity_error,
  }


def _com_to_link_velocities(data: dict[str, np.ndarray], model) -> None:
  """Convert the HOPE COM velocity field to mjlab's link-origin convention."""
  names = [str(name) for name in data["body_names"].tolist()]
  body_ids = [model.body(name).id for name in names]
  frames = len(data["body_pos_w"])
  rotations = Rotation.from_quat(data["body_quat_w"].reshape(-1, 4), scalar_first=True)
  inertial_offset = np.broadcast_to(
    model.body_ipos[body_ids], (frames, len(body_ids), 3)
  )
  inertial_offset_w = rotations.apply(inertial_offset.reshape(-1, 3)).reshape(
    inertial_offset.shape
  )
  data["body_lin_vel_w"] = (
    data["body_lin_vel_w"] + np.cross(data["body_ang_vel_w"], -inertial_offset_w)
  ).astype(np.float32)
  if "root_link_lin_vel_w" in data:
    np.testing.assert_allclose(
      data["body_lin_vel_w"][:, names.index(ROOT_BODY)],
      data["root_link_lin_vel_w"],
      atol=1e-6,
    )


def _verify_asset_contract(source_manifest: dict, xml: Path) -> Path:
  asset_manifest = xml.with_name("asset_manifest.json")
  if sha256(xml) != source_manifest["model"]["sha256"]:
    raise ValueError("Installed G1 MJCF differs from the motion retarget model.")
  if sha256(asset_manifest) != source_manifest["asset_manifest_sha256"]:
    raise ValueError("Installed G1 asset manifest differs from the motion source.")
  return asset_manifest


def prepare(
  source: Path,
  output: Path,
  source_manifest_path: Path,
  xml: Path,
) -> dict:
  if source.resolve() == output.resolve():
    raise ValueError("Source and output must differ so raw clips stay unchanged.")
  source_manifest, sources = _verify_sources(source, source_manifest_path)
  asset_manifest = _verify_asset_contract(source_manifest, xml)
  source_velocity_convention = _source_velocity_convention(source_manifest)
  model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
  records = {Path(record["path"]).name: record for record in source_manifest["clips"]}
  output.mkdir(parents=True, exist_ok=True)
  entries = []
  outputs = []
  for source_file in sources:
    record = records[source_file.name]
    with np.load(source_file, allow_pickle=False) as archive:
      data = {key: np.array(value, copy=True) for key, value in archive.items()}
      source_validation = _validate_source_archive(
        data, model, source_manifest, record, source_file
      )
      _com_to_link_velocities(data, model)
      result, summary = canonicalize_motion(
        data,
        strike_frame=int(record["strike_frame"]),
        root_body=ROOT_BODY,
        wrist_body=WRIST_BODY,
        racket_offset=RACKET_OFFSET,
        racket_normal=RACKET_NORMAL,
        feet_bodies=FEET_BODIES,
      )
    # This source-only MuJoCo qvel vector is in the raw world frame. Core
    # training consumes the named joint/body fields, so omit the stale copy.
    result.pop("generalized_velocity", None)
    result["strike_frame"] = np.asarray(record["strike_frame"], dtype=np.int64)
    strike_time_s = float(
      record.get(
        "strike_time_s",
        record["strike_frame"] / int(record.get("output_fps", source_manifest["fps"])),
      )
    )
    result["strike_time_s"] = np.asarray(strike_time_s, dtype=np.float64)
    result["body_velocity_convention"] = np.asarray(
      "link-origin world linear/angular velocity"
    )
    target = output / source_file.name
    # NumPy's stub treats every expanded value as a possible allow_pickle arg.
    np.savez_compressed(target, **result)  # ty: ignore[invalid-argument-type]
    summary.update(
      file=source_file.name,
      hit_type=record["hit_type"],
      source_sha256=record["sha256"],
      output_sha256=sha256(target),
      source_bvh_sha256=record["source_bvh"]["sha256"],
      source_validation=source_validation,
    )
    entries.append(summary)
    outputs.append(target)
  manifest = {
    "format_version": 1,
    "source_manifest": str(source_manifest_path),
    "source_manifest_sha256": sha256(source_manifest_path),
    "source_retarget_method": source_manifest["retarget_method"],
    "source_independent_acceptance": source_manifest["independent_acceptance"],
    "model": {"path": str(xml), "sha256": sha256(xml)},
    "asset_manifest": {
      "path": str(asset_manifest),
      "sha256": sha256(asset_manifest),
    },
    "motion_directory": str(output),
    "num_clips": len(entries),
    "total_frames": sum(entry["frames"] for entry in entries),
    "total_duration_s": sum(entry["duration_s"] for entry in entries),
    "joint_names": source_manifest["joint_names"],
    "body_names": source_manifest["body_names"],
    "quaternion_order": "wxyz",
    "velocity_frame": "world",
    "source_body_velocity_convention": source_velocity_convention,
    "training_body_velocity_convention": (
      "link-origin world linear/angular velocity, converted with MJCF inertial offsets"
    ),
    "canonical_frame": (
      "initial pelvis XY=(0,0), initial pelvis yaw=0, original ground Z"
    ),
    "strike_annotation": "HOPE strike frame 43 / 0.86 s",
    "clips": entries,
    "fk_validation": validate_fk(outputs, xml, root_body=ROOT_BODY),
  }
  destination = output.parent / "manifest.json"
  destination.write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  args = parser.parse_args()
  manifest = prepare(args.source, args.output, args.source_manifest, args.xml)
  print(
    f"Prepared {manifest['num_clips']} clips / {manifest['total_frames']} frames "
    f"/ {manifest['total_duration_s']:.2f} s in {args.output}"
  )
  print(json.dumps(manifest["fk_validation"], indent=2))


if __name__ == "__main__":
  main()

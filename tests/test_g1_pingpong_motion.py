"""Contracts for the canonical 29-DoF G1 ping-pong motion corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_BODY_NAMES,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_ROOT_BODY,
  G1_PINGPONG_XML,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "dataset" / "g1_pingpong"
RAW_DIR = DATA_DIR / "raw"
MOTION_DIR = DATA_DIR / "motions"
SOURCE_MANIFEST = DATA_DIR / "source_manifest.json"
TRAINING_MANIFEST = DATA_DIR / "manifest.json"
TELE_GMR_DATA_DIR = REPO_ROOT / "dataset" / "g1_pingpong_tele_gmr"
TELE_GMR_RAW_DIR = TELE_GMR_DATA_DIR / "raw"
TELE_GMR_MOTION_DIR = TELE_GMR_DATA_DIR / "motions"
TELE_GMR_SOURCE_MANIFEST = TELE_GMR_DATA_DIR / "source_manifest.json"
TELE_GMR_TRAINING_MANIFEST = TELE_GMR_DATA_DIR / "manifest.json"

EXPECTED_CLIPS = 43
EXPECTED_FRAMES = 94
EXPECTED_FPS = 50
EXPECTED_STRIKE_FRAME = 43
EXPECTED_STRIKE_TIME_S = 0.86
FK_TOLERANCE = 1.0e-6
VELOCITY_TOLERANCE = 2.0e-6

pytestmark = pytest.mark.skipif(
  not RAW_DIR.is_dir() or not any(MOTION_DIR.glob("*.npz")),
  reason="Optional G1 reference datasets are not installed.",
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
  return json.loads(path.read_text())


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
  w, x, y, z = quaternion
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_about_z(vectors: np.ndarray, angle: float) -> np.ndarray:
  cosine, sine = math.cos(angle), math.sin(angle)
  rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
  return vectors @ rotation.T


def _write_raw_state_to_mujoco(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  archive: np.lib.npyio.NpzFile,
  frame: int,
) -> None:
  body_names = tuple(str(name) for name in archive["body_names"])
  joint_names = tuple(str(name) for name in archive["joint_names"])
  root_index = body_names.index(G1_PINGPONG_ROOT_BODY)
  joint_qpos_addresses = [
    int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names
  ]

  data.qpos[:] = model.qpos0
  data.qpos[:3] = archive["body_pos_w"][frame, root_index]
  data.qpos[3:7] = archive["body_quat_w"][frame, root_index]
  data.qpos[joint_qpos_addresses] = archive["joint_pos"][frame]
  data.qvel[:] = archive["generalized_velocity"][frame]
  mujoco.mj_forward(model, data)


def test_g1_pingpong_motion_schema_and_hashes():
  source_manifest = _load_json(SOURCE_MANIFEST)
  training_manifest = _load_json(TRAINING_MANIFEST)
  raw_files = sorted(RAW_DIR.glob("*.npz"))
  motion_files = sorted(MOTION_DIR.glob("*.npz"))

  assert len(raw_files) == EXPECTED_CLIPS
  assert len(motion_files) == EXPECTED_CLIPS
  assert source_manifest["num_motions"] == EXPECTED_CLIPS
  assert training_manifest["num_clips"] == EXPECTED_CLIPS
  assert training_manifest["source_manifest_sha256"] == _sha256(SOURCE_MANIFEST)
  assert tuple(training_manifest["joint_names"]) == G1_PINGPONG_JOINT_NAMES
  assert tuple(training_manifest["body_names"]) == G1_PINGPONG_BODY_NAMES
  assert len(G1_PINGPONG_JOINT_NAMES) == 29
  assert len(G1_PINGPONG_BODY_NAMES) == 30

  source_records = {
    Path(record["path"]).name: record for record in source_manifest["clips"]
  }
  output_records = {record["file"]: record for record in training_manifest["clips"]}
  expected_names = set(source_records)
  assert {path.name for path in raw_files} == expected_names
  assert {path.name for path in motion_files} == expected_names
  assert set(output_records) == expected_names

  for raw_path, motion_path in zip(raw_files, motion_files, strict=True):
    assert raw_path.name == motion_path.name
    assert _sha256(raw_path) == source_records[raw_path.name]["sha256"]
    assert _sha256(motion_path) == output_records[motion_path.name]["output_sha256"]

    with np.load(motion_path, allow_pickle=False) as archive:
      assert archive["joint_names"].dtype.kind == "U"
      assert archive["body_names"].dtype.kind == "U"
      assert archive["body_velocity_convention"].dtype.kind == "U"
      assert tuple(str(name) for name in archive["joint_names"]) == (
        G1_PINGPONG_JOINT_NAMES
      )
      assert tuple(str(name) for name in archive["body_names"]) == (
        G1_PINGPONG_BODY_NAMES
      )
      assert archive["fps"].shape == (1,)
      assert int(archive["fps"][0]) == EXPECTED_FPS
      assert archive["joint_pos"].shape == (EXPECTED_FRAMES, 29)
      assert archive["joint_vel"].shape == (EXPECTED_FRAMES, 29)
      assert archive["body_pos_w"].shape == (EXPECTED_FRAMES, 30, 3)
      assert archive["body_quat_w"].shape == (EXPECTED_FRAMES, 30, 4)
      assert archive["body_lin_vel_w"].shape == (EXPECTED_FRAMES, 30, 3)
      assert archive["body_ang_vel_w"].shape == (EXPECTED_FRAMES, 30, 3)
      assert archive["strike_frame"].shape == ()
      assert archive["strike_time_s"].shape == ()
      assert int(archive["strike_frame"]) == EXPECTED_STRIKE_FRAME
      assert float(archive["strike_time_s"]) == EXPECTED_STRIKE_TIME_S
      assert int(archive["hope_strike_frame"][0]) == EXPECTED_STRIKE_FRAME
      assert "generalized_velocity" not in archive.files
      assert str(archive["body_velocity_convention"]) == (
        "link-origin world linear/angular velocity"
      )

      for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "racket_pos_w",
        "racket_lin_vel_w",
        "racket_normal_w",
      ):
        assert np.isfinite(archive[key]).all(), (motion_path.name, key)
      quaternion_norms = np.linalg.norm(archive["body_quat_w"], axis=-1)
      np.testing.assert_allclose(quaternion_norms, 1.0, atol=1.0e-6)


def test_g1_pingpong_canonical_root_and_fk_contract():
  manifest = _load_json(TRAINING_MANIFEST)
  recorded_fk = manifest["fk_validation"]
  assert recorded_fk["frames_checked"] == EXPECTED_CLIPS * 3
  assert recorded_fk["max_body_position_error_m"] <= FK_TOLERANCE
  assert recorded_fk["max_non_ankle_position_error_m"] <= FK_TOLERANCE
  assert recorded_fk["max_body_quaternion_l2_error"] <= FK_TOLERANCE

  model = mujoco.MjModel.from_xml_path(str(G1_PINGPONG_XML))
  data = mujoco.MjData(model)
  body_ids = [model.body(name).id for name in G1_PINGPONG_BODY_NAMES]
  joint_qpos_addresses = [
    int(model.jnt_qposadr[model.joint(name).id]) for name in G1_PINGPONG_JOINT_NAMES
  ]
  root_index = G1_PINGPONG_BODY_NAMES.index(G1_PINGPONG_ROOT_BODY)
  non_ankle = np.array(
    ["ankle" not in name for name in G1_PINGPONG_BODY_NAMES], dtype=bool
  )
  max_position_error = 0.0
  max_non_ankle_error = 0.0
  max_quaternion_error = 0.0

  for path in sorted(MOTION_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as archive:
      initial_root_position = archive["body_pos_w"][0, root_index]
      np.testing.assert_allclose(initial_root_position[:2], 0.0, atol=1.0e-7)
      assert abs(_yaw_from_wxyz(archive["body_quat_w"][0, root_index])) <= 1.0e-7

      frames = (0, int(archive["strike_frame"]), EXPECTED_FRAMES - 1)
      for frame in frames:
        data.qpos[:] = model.qpos0
        data.qpos[:3] = archive["body_pos_w"][frame, root_index]
        data.qpos[3:7] = archive["body_quat_w"][frame, root_index]
        data.qpos[joint_qpos_addresses] = archive["joint_pos"][frame]
        mujoco.mj_forward(model, data)

        position_errors = np.linalg.norm(
          data.xpos[body_ids] - archive["body_pos_w"][frame], axis=-1
        )
        expected_quaternions = archive["body_quat_w"][frame]
        quaternion_errors = np.minimum(
          np.linalg.norm(data.xquat[body_ids] - expected_quaternions, axis=-1),
          np.linalg.norm(data.xquat[body_ids] + expected_quaternions, axis=-1),
        )
        max_position_error = max(max_position_error, float(position_errors.max()))
        max_non_ankle_error = max(
          max_non_ankle_error, float(position_errors[non_ankle].max())
        )
        max_quaternion_error = max(max_quaternion_error, float(quaternion_errors.max()))

  assert max_position_error <= FK_TOLERANCE
  assert max_non_ankle_error <= FK_TOLERANCE
  assert max_quaternion_error <= FK_TOLERANCE


def test_g1_pingpong_processed_link_and_racket_site_velocity_contract():
  raw_files = sorted(RAW_DIR.glob("*.npz"))
  sampled_files = (raw_files[0], raw_files[len(raw_files) // 2], raw_files[-1])
  sampled_frames = (0, EXPECTED_STRIKE_FRAME, EXPECTED_FRAMES - 1)
  model = mujoco.MjModel.from_xml_path(str(G1_PINGPONG_XML))
  data = mujoco.MjData(model)
  site_id = model.site(G1_PINGPONG_RACKET_SITE).id
  maximum_source_com_to_link_gap = 0.0

  for raw_path in sampled_files:
    processed_path = MOTION_DIR / raw_path.name
    with (
      np.load(raw_path, allow_pickle=False) as raw,
      np.load(processed_path, allow_pickle=False) as processed,
    ):
      body_names = tuple(str(name) for name in raw["body_names"])
      body_ids = [model.body(name).id for name in body_names]
      yaw_correction = float(processed["canonical_initial_yaw_correction_rad"])

      for frame in sampled_frames:
        _write_raw_state_to_mujoco(model, data, raw, frame)
        link_velocities = []
        for body_id in body_ids:
          jacobian_position = np.zeros((3, model.nv))
          jacobian_rotation = np.zeros((3, model.nv))
          mujoco.mj_jacBody(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            body_id,
          )
          link_velocities.append(jacobian_position @ data.qvel)
        link_velocities = np.asarray(link_velocities)
        maximum_source_com_to_link_gap = max(
          maximum_source_com_to_link_gap,
          float(
            np.linalg.norm(
              raw["body_lin_vel_w"][frame] - link_velocities, axis=-1
            ).max()
          ),
        )
        expected_processed_link_velocities = _rotate_about_z(
          link_velocities, yaw_correction
        )
        np.testing.assert_allclose(
          processed["body_lin_vel_w"][frame],
          expected_processed_link_velocities,
          atol=VELOCITY_TOLERANCE,
          rtol=1.0e-6,
        )

        site_jacobian_position = np.zeros((3, model.nv))
        site_jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
          model,
          data,
          site_jacobian_position,
          site_jacobian_rotation,
          site_id,
        )
        expected_site_velocity = _rotate_about_z(
          site_jacobian_position @ data.qvel, yaw_correction
        )
        np.testing.assert_allclose(
          processed["racket_lin_vel_w"][frame],
          expected_site_velocity,
          atol=VELOCITY_TOLERANCE,
          rtol=1.0e-6,
        )

  # Prove this test distinguishes the source COM convention from the processed
  # link-origin convention rather than accepting an unchanged source field.
  assert maximum_source_com_to_link_gap > 0.1


def test_g1_tele_gmr_motion_provenance_hashes_fk_and_velocity_contract():
  source_manifest = _load_json(TELE_GMR_SOURCE_MANIFEST)
  training_manifest = _load_json(TELE_GMR_TRAINING_MANIFEST)
  assert _sha256(TELE_GMR_SOURCE_MANIFEST) == (
    "87f402472ec33317c93a5a921b386f27f487d1bd6259bdaf7065e53be1b8f690"
  )
  assert source_manifest["retarget_method"] == "tele_gmr_football"
  assert source_manifest["asset_manifest_sha256"] == (
    "694163f5f6c425a0c1f1e6a09df5f1db644688ce6d3d3da3dd90d4c699e0ac2a"
  )
  assert source_manifest["independent_acceptance"] == {
    "path": (
      "/home/lxz/HOPE/hope_training/mjlab_pingpong/evaluations/"
      "g1_tele_gmr_football_acceptance_20261003.json"
    ),
    "sha256": "570b5eb7504c8a832b696d852b6ae17b70b2d2819bcde21bb3c917b7f33082ee",
    "all_passed": True,
    "passed_clips": EXPECTED_CLIPS,
  }
  assert training_manifest["source_manifest_sha256"] == _sha256(
    TELE_GMR_SOURCE_MANIFEST
  )
  assert training_manifest["source_retarget_method"] == "tele_gmr_football"
  assert training_manifest["num_clips"] == EXPECTED_CLIPS
  assert training_manifest["total_frames"] == EXPECTED_CLIPS * EXPECTED_FRAMES
  assert training_manifest["fk_validation"]["max_body_position_error_m"] < 1.0e-6

  source_records = {
    Path(record["path"]).name: record for record in source_manifest["clips"]
  }
  output_records = {record["file"]: record for record in training_manifest["clips"]}
  raw_files = sorted(TELE_GMR_RAW_DIR.glob("*.npz"))
  motion_files = sorted(TELE_GMR_MOTION_DIR.glob("*.npz"))
  assert len(raw_files) == len(motion_files) == EXPECTED_CLIPS
  assert set(source_records) == set(output_records) == {p.name for p in raw_files}
  assert {p.name for p in motion_files} == set(source_records)
  for raw_path, motion_path in zip(raw_files, motion_files, strict=True):
    assert raw_path.name == motion_path.name
    assert _sha256(raw_path) == source_records[raw_path.name]["sha256"]
    assert _sha256(motion_path) == output_records[motion_path.name]["output_sha256"]

  model = mujoco.MjModel.from_xml_path(str(G1_PINGPONG_XML))
  data = mujoco.MjData(model)
  wrist_id = model.body("right_wrist_yaw_link").id
  wrist_index = G1_PINGPONG_BODY_NAMES.index("right_wrist_yaw_link")
  for raw_path in (raw_files[0], raw_files[len(raw_files) // 2], raw_files[-1]):
    with (
      np.load(raw_path, allow_pickle=False) as raw,
      np.load(TELE_GMR_MOTION_DIR / raw_path.name, allow_pickle=False) as processed,
    ):
      yaw_correction = float(processed["canonical_initial_yaw_correction_rad"])
      for frame in (0, EXPECTED_STRIKE_FRAME, EXPECTED_FRAMES - 1):
        _write_raw_state_to_mujoco(model, data, raw, frame)
        com_jacobian = np.zeros((3, model.nv))
        link_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, com_jacobian, rotation_jacobian, wrist_id)
        np.testing.assert_allclose(
          raw["body_lin_vel_w"][frame, wrist_index],
          com_jacobian @ data.qvel,
          atol=VELOCITY_TOLERANCE,
          rtol=1.0e-6,
        )
        mujoco.mj_jacBody(model, data, link_jacobian, rotation_jacobian, wrist_id)
        np.testing.assert_allclose(
          processed["body_lin_vel_w"][frame, wrist_index],
          _rotate_about_z(link_jacobian @ data.qvel, yaw_correction),
          atol=VELOCITY_TOLERANCE,
          rtol=1.0e-6,
        )

"""CPU checks for the supplied A3 motion coordinate conversion."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/tools/prepare_a3_motion.py"
spec = importlib.util.spec_from_file_location("prepare_a3_motion", MODULE_PATH)
assert spec is not None and spec.loader is not None
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def make_motion():
  frames = 5
  quaternions = np.tile(
    Rotation.from_euler("z", -90, degrees=True).as_quat(scalar_first=True),
    (frames, 2, 1),
  )
  return {
    "fps": np.array([50]),
    "joint_names": np.array(["right_wrist_yaw_joint"]),
    "body_names": np.array(["pelvis_link", "right_wrist_yaw_Link"]),
    "joint_pos": np.zeros((frames, 1)),
    "joint_vel": np.zeros((frames, 1)),
    "body_pos_w": np.tile([[2, 3, 1], [3, 3, 1]], (frames, 1, 1)).astype(float),
    "body_quat_w": quaternions,
    "body_lin_vel_w": np.tile([1, 0, 0], (frames, 2, 1)).astype(float),
    "body_ang_vel_w": np.tile([0, 0, 2], (frames, 2, 1)).astype(float),
  }


def test_a3_motion_changes_all_world_fields_consistently():
  raw = make_motion()
  result, summary = prepare.canonicalize_motion(raw, strike_frame=2)
  np.testing.assert_allclose(result["body_pos_w"][0], [[0, 0, 1], [0, 1, 1]], atol=1e-6)
  np.testing.assert_allclose(result["body_quat_w"][0, 0], [1, 0, 0, 0], atol=1e-6)
  np.testing.assert_allclose(result["body_lin_vel_w"][0, 0], [0, 1, 0], atol=1e-6)
  np.testing.assert_allclose(result["body_ang_vel_w"][0, 0], [0, 0, 2], atol=1e-6)
  np.testing.assert_allclose(
    result["racket_pos_w"][0], [0, 1, 1] + prepare.RACKET_OFFSET, atol=1e-6
  )
  expected_velocity = [0, 1, 0] + np.cross([0, 0, 2], prepare.RACKET_OFFSET)
  np.testing.assert_allclose(
    result["racket_lin_vel_w"][0], expected_velocity, atol=1e-6
  )
  np.testing.assert_allclose(result["racket_normal_w"][0], [0, 1, 0], atol=1e-6)
  np.testing.assert_array_equal(raw["body_pos_w"][0, 0], [2, 3, 1])
  assert summary["duration_s"] == 0.08
  assert result["strike_time_s"] == 0.04


@pytest.mark.parametrize("invalid", ["nonfinite", "quaternion", "duplicate", "fps"])
def test_a3_motion_rejects_invalid_data(invalid):
  raw = make_motion()
  if invalid == "nonfinite":
    raw["joint_pos"][0, 0] = np.nan
  elif invalid == "quaternion":
    raw["body_quat_w"][0, 0] = 0
  elif invalid == "duplicate":
    raw["body_names"][1] = raw["body_names"][0]
  else:
    raw["fps"] = np.array([0])
  with pytest.raises(ValueError):
    prepare.canonicalize_motion(raw, strike_frame=2)


def test_a3_motion_corpus_preserves_names_and_canonical_root():
  root = Path(__file__).resolve().parents[1] / "dataset/a3_pingpong"
  files = sorted((root / "motions").glob("*.npz"))
  if not files:
    pytest.skip("Local A3 dataset not installed.")
  assert len(files) == 61
  for path in files:
    with np.load(path, allow_pickle=False) as data:
      prepare.validate_motion(dict(data))
      root_idx = list(data["body_names"]).index("pelvis_link")
      np.testing.assert_allclose(data["body_pos_w"][0, root_idx, :2], 0, atol=1e-6)
      yaw = Rotation.from_quat(
        data["body_quat_w"][0, root_idx], scalar_first=True
      ).as_euler("xyz")[2]
      assert abs(yaw) < 1e-6
      assert data["joint_pos"].shape == (94, 31)
      assert float(data["strike_time_s"]) == 0.86

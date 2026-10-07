"""Regression checks for the nameless G1 motion archive schema."""

import numpy as np
import torch

from mjlab.tasks.adapt_tennis.mdp.commands import MotionLoader


LEGACY_BODIES = (
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
LEGACY_JOINTS = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)


def test_nameless_legacy_columns_are_remapped_by_schema(tmp_path):
  path = tmp_path / "legacy.npz"
  body = np.arange(30 * 3, dtype=np.float32).reshape(1, 30, 3)
  quat = np.zeros((1, 30, 4), dtype=np.float32)
  quat[..., 0] = 1
  joint = np.arange(29, dtype=np.float32).reshape(1, 29)
  np.savez(
    path,
    fps=np.asarray([50], dtype=np.int64),
    body_pos_w=body,
    body_quat_w=quat,
    body_lin_vel_w=np.zeros_like(body),
    body_ang_vel_w=np.zeros_like(body),
    joint_pos=joint,
    joint_vel=np.zeros_like(joint),
  )

  target_bodies = ("pelvis", "torso_link", "left_ankle_roll_link")
  target_joints = ("left_hip_pitch_joint", "waist_pitch_joint", "right_wrist_yaw_joint")
  loader = MotionLoader(
    str(path),
    torch.tensor([0, 9, 18]),
    device="cpu",
    target_body_names=target_bodies,
    target_joint_names=target_joints,
    legacy_body_names=LEGACY_BODIES,
    legacy_joint_names=LEGACY_JOINTS,
    align_heading_to_frame="none",
  )

  assert loader.body_pos_w[0, :, 0].tolist() == [0.0, 27.0, 54.0]
  # The source order is BFS; target joint names must be looked up in it.
  assert loader.joint_pos[0].tolist() == [0.0, 8.0, 28.0]

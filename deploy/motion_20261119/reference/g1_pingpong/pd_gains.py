"""Manually specified execution PD, shared by MuJoCo and real deployment.

Edit this table to change base gains. Order is resolved by joint name.
Columns: ping-pong Kp, ping-pong Kd, startup/AMP Kp, startup/AMP Kd.
Initial literals preserve the 2026-10-15 baseline; no training/XML/metadata
gain is consulted at runtime. Action scaling is deliberately independent.
"""

import numpy as np

JOINT_PD = {
  "left_hip_pitch_joint": (40.17923863450712, 2.557889775413375, 100.0, 2.0),
  "left_hip_roll_joint": (99.09842777666111, 6.308801853496639, 100.0, 2.0),
  "left_hip_yaw_joint": (40.17923863450712, 2.557889775413375, 100.0, 2.0),
  "left_knee_joint": (99.09842777666111, 6.308801853496639, 150.0, 4.0),
  "left_ankle_pitch_joint": (28.50124619574858, 1.814445686584846, 40.0, 2.0),
  "left_ankle_roll_joint": (28.50124619574858, 1.814445686584846, 40.0, 2.0),
  "right_hip_pitch_joint": (40.17923863450712, 2.557889775413375, 100.0, 2.0),
  "right_hip_roll_joint": (99.09842777666111, 6.308801853496639, 100.0, 2.0),
  "right_hip_yaw_joint": (40.17923863450712, 2.557889775413375, 100.0, 2.0),
  "right_knee_joint": (99.09842777666111, 6.308801853496639, 150.0, 4.0),
  "right_ankle_pitch_joint": (28.50124619574858, 1.814445686584846, 40.0, 2.0),
  "right_ankle_roll_joint": (28.50124619574858, 1.814445686584846, 40.0, 2.0),
  "waist_yaw_joint": (40.17923863450712, 2.557889775413375, 400.0, 5.0),
  "waist_roll_joint": (28.50124619574858, 1.814445686584846, 400.0, 5.0),
  "waist_pitch_joint": (28.50124619574858, 1.814445686584846, 400.0, 5.0),
  "left_shoulder_pitch_joint": (14.25062309787429, 0.907222843292423, 100.0, 2.0),
  "left_shoulder_roll_joint": (14.25062309787429, 0.907222843292423, 100.0, 2.0),
  "left_shoulder_yaw_joint": (14.25062309787429, 0.907222843292423, 50.0, 2.0),
  "left_elbow_joint": (14.25062309787429, 0.907222843292423, 50.0, 2.0),
  "left_wrist_roll_joint": (14.25062309787429, 0.907222843292423, 20.0, 1.0),
  "left_wrist_pitch_joint": (16.77832748089279, 1.06814150219, 20.0, 1.0),
  "left_wrist_yaw_joint": (16.77832748089279, 1.06814150219, 20.0, 1.0),
  "right_shoulder_pitch_joint": (14.25062309787429, 0.907222843292423, 100.0, 2.0),
  "right_shoulder_roll_joint": (14.25062309787429, 0.907222843292423, 100.0, 2.0),
  "right_shoulder_yaw_joint": (14.25062309787429, 0.907222843292423, 50.0, 2.0),
  "right_elbow_joint": (14.25062309787429, 0.907222843292423, 50.0, 2.0),
  "right_wrist_roll_joint": (14.25062309787429, 0.907222843292423, 20.0, 1.0),
  "right_wrist_pitch_joint": (16.77832748089279, 1.06814150219, 20.0, 1.0),
  "right_wrist_yaw_joint": (16.77832748089279, 1.06814150219, 20.0, 1.0),
}


def deployment_gains(joint_names, *, profile="pingpong", scales=None):
  """Return independent gain arrays; scales never alter the literal base table."""
  names = list(joint_names)
  if (
    not names or len(set(names)) != len(names) or any(n not in JOINT_PD for n in names)
  ):
    raise ValueError("Unknown or duplicate joint in manual deployment PD")
  if profile not in ("pingpong", "stable"):
    raise ValueError("Unknown deployment PD profile")
  options = scales or {}
  offset = 0 if profile == "pingpong" else 2
  gains = []
  for i, kind in enumerate(("kp", "kd")):
    scale = options.get(f"{kind}_scale", 1.0)
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
      raise ValueError("Deployment PD scales must be finite and positive")
    overrides = options.get(f"{kind}_joint_scales", {})
    if not isinstance(overrides, dict) or any(n not in names for n in overrides):
      raise ValueError("Unknown joint in deployment PD scale overrides")
    factors = np.array([overrides.get(n, scale) for n in names], dtype=float)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
      raise ValueError("Joint PD scales must be finite and positive")
    gains.append(np.array([JOINT_PD[n][offset + i] for n in names]) * factors)
  return tuple(gains)

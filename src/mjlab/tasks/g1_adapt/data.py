"""Clip-level splits and labeled G1 sequences; no cross-clip transitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from .motion import pack_state, quaternion_6d, rotation_6d
from .mvae import file_hash


def load_clip(path, fps=50.0, labels=None):
  with np.load(path, allow_pickle=True) as archive:
    data = {key: archive[key] for key in archive.files}
  if abs(float(np.asarray(data["fps"]).reshape(-1)[0]) - fps) > 1e-5:
    raise ValueError(f"{path}: resample motion to {fps} Hz before MVAE training")
  if tuple(map(str, data.get("joint_names", ()))) != G1_PINGPONG_JOINT_NAMES:
    raise ValueError(f"{path}: MVAE requires explicit canonical joint_names")
  names = list(map(str, data["body_names"]))
  root = names.index("pelvis")
  joints = torch.as_tensor(data["joint_pos"], dtype=torch.float32)
  count = len(joints)
  if count < 3:
    raise ValueError(f"{path}: motion too short")
  annotation = (labels or {}).get(Path(path).stem, {})
  hit = int(
    annotation.get(
      "strike_frame",
      np.asarray(data.get("strike_frame", round(0.46 * (count - 1)))).reshape(-1)[0],
    )
  )
  if not 0 < hit < count - 1:
    raise ValueError(f"{path}: invalid strike frame")
  index = torch.arange(count, dtype=torch.float32)
  angle = torch.where(
    index <= hit,
    index / hit * torch.pi,
    torch.pi + (index - hit) / (count - 1 - hit) * torch.pi,
  )
  phase = torch.stack((angle.cos(), angle.sin()), -1)
  state = pack_state(
    joints,
    torch.as_tensor(data["joint_vel"]).float(),
    torch.as_tensor(data["body_quat_w"][:, root]).float(),
    torch.as_tensor(data["body_lin_vel_w"][:, root]).float(),
    torch.as_tensor(data["body_ang_vel_w"][:, root]).float(),
    phase,
  )
  stroke_name = str(
    annotation.get("stroke", np.asarray(data.get("hit_type", "unknown")).reshape(-1)[0])
  )
  stroke = {"forehand": 0, "backhand": 1}.get(stroke_name, -1)
  spin = {"topspin": 0, "flat": 0, "slice": 1}.get(
    str(annotation.get("spin", "unknown")), -1
  )
  if not torch.isfinite(state).all():
    raise ValueError(f"{path}: non-finite motion")
  return {
    "state": state,
    "stroke": stroke,
    "spin": spin,
    "path": str(path),
    "corrected": bool(
      np.asarray(data.get("adapt_physics_corrected", False)).reshape(-1)[0]
    ),
    "sha256": file_hash(path),
  }


def load_sequences(directory, labels_file=None, fps=50.0):
  paths = sorted(Path(directory).expanduser().rglob("*.npz"))
  if not paths:
    raise FileNotFoundError(f"No motion NPZ in {directory}")
  labels = json.loads(Path(labels_file).read_text()) if labels_file else {}
  return [load_clip(path, fps, labels) for path in paths]


def sample_windows(clips, batch, history, rollout, device, yaw_augmentation=True):
  eligible = [clip for clip in clips if len(clip["state"]) >= history + rollout]
  if not eligible:
    raise ValueError("No clip long enough for history + rollout")
  windows, strokes, spins = [], [], []
  for clip_index in torch.randint(len(eligible), (batch,)).tolist():
    clip = eligible[clip_index]
    start = int(torch.randint(len(clip["state"]) - history - rollout + 1, (1,)))
    windows.append(clip["state"][start : start + history + rollout])
    strokes.append(clip["stroke"])
    spins.append(clip["spin"])
  states = torch.stack(windows).to(device)
  if yaw_augmentation:
    yaw = (torch.rand(batch, device=device) - 0.5) * (torch.pi / 6)
    turn = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)[
      :, None
    ].expand(-1, history + rollout, -1)
    states[..., :3] = quat_apply(turn, states[..., :3])
    states[..., 3:6] = quat_apply(turn, states[..., 3:6])
    states[..., 6:12] = rotation_6d(quat_mul(turn, quaternion_6d(states[..., 6:12])))
  return (
    states,
    torch.tensor(strokes, device=device),
    torch.tensor(spins, device=device),
  )

"""Unitree G1 29-DoF ping-pong asset and position-servo configuration.

This package is intentionally separate from :mod:`unitree_g1`, whose racket
models expose a different 27-DoF articulation. See ``README.md`` for asset
provenance and the validation performed during conversion from URDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

CollisionMode = Literal["mesh", "primitive"]

ASSET_DIR = Path(__file__).resolve().parent
G1_PINGPONG_XML = ASSET_DIR / "g1.xml"
G1_PINGPONG_SOURCE_URDF = ASSET_DIR / "source" / "g1_29dof_pingpong.urdf"
G1_PINGPONG_ASSET_MANIFEST = ASSET_DIR / "asset_manifest.json"
G1_PINGPONG_ASSET_VALIDATION = ASSET_DIR / "asset_validation.json"

with G1_PINGPONG_ASSET_MANIFEST.open() as manifest_file:
  MANIFEST = json.load(manifest_file)

# Native MuJoCo kinematic-tree order. The servo/action order is identical.
G1_PINGPONG_JOINT_NAMES: tuple[str, ...] = tuple(MANIFEST["mesh"]["joint_names"])
G1_PINGPONG_BODY_NAMES: tuple[str, ...] = tuple(MANIFEST["mesh"]["body_names"])
G1_PINGPONG_COLLISION_GEOMS_BY_BODY: dict[str, tuple[str, ...]] = {
  name: tuple(geoms) for name, geoms in MANIFEST["mesh"]["collision_geoms"].items()
}

G1_PINGPONG_ROOT_BODY = "pelvis"
G1_PINGPONG_ANCHOR_BODY = "torso_link"
G1_PINGPONG_FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_PINGPONG_WRIST_BODY = "right_wrist_yaw_link"
G1_PINGPONG_HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
G1_PINGPONG_TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
G1_PINGPONG_RACKET_BODY = MANIFEST["racket"]["body"]
G1_PINGPONG_RACKET_SITE = MANIFEST["racket"]["site"]
G1_PINGPONG_MOUNT_OFFSET = tuple(MANIFEST["racket"]["pos"])
G1_PINGPONG_MOUNT_QUAT = tuple(MANIFEST["racket"]["quat_wxyz"])
G1_PINGPONG_RACKET_NORMAL = tuple(MANIFEST["racket"]["normal"])

G1_PINGPONG_JOINT_VELOCITY_LIMITS: dict[str, float] = dict(MANIFEST["velocity_limits"])
G1_PINGPONG_JOINT_EFFORT_LIMITS: dict[str, float] = dict(MANIFEST["effort_limits"])
G1_PINGPONG_ACTION_SCALE: dict[str, float] = {
  name: settings["action_scale"]
  for name, settings in MANIFEST["actuator_settings"].items()
}

# Source-compatible names ease migration from the archived HOPE module while
# keeping the package itself separate from the existing 27-DoF G1 package.
JOINT_NAMES = G1_PINGPONG_JOINT_NAMES
MUJOCO_JOINT_NAMES = G1_PINGPONG_JOINT_NAMES
BODY_NAMES = G1_PINGPONG_BODY_NAMES
ISAAC_BODY_NAMES = G1_PINGPONG_BODY_NAMES
COLLISION_GEOMS_BY_BODY = G1_PINGPONG_COLLISION_GEOMS_BY_BODY
PRIMITIVE_COLLISION_GEOMS_BY_BODY = G1_PINGPONG_COLLISION_GEOMS_BY_BODY
ROOT_BODY = G1_PINGPONG_ROOT_BODY
ANCHOR_BODY = G1_PINGPONG_ANCHOR_BODY
FEET_BODIES = G1_PINGPONG_FEET_BODIES
WRIST_BODY = G1_PINGPONG_WRIST_BODY
MOUNT_OFFSET = G1_PINGPONG_MOUNT_OFFSET
MOUNT_QUAT = G1_PINGPONG_MOUNT_QUAT
RACKET_NORMAL = G1_PINGPONG_RACKET_NORMAL
RACKET_SITE = G1_PINGPONG_RACKET_SITE
JOINT_VELOCITY_LIMITS = G1_PINGPONG_JOINT_VELOCITY_LIMITS
JOINT_EFFORT_LIMITS = G1_PINGPONG_JOINT_EFFORT_LIMITS
ACTION_SCALE = G1_PINGPONG_ACTION_SCALE
G1_ACTION_SCALE = G1_PINGPONG_ACTION_SCALE


def get_spec(collision_mode: CollisionMode = "mesh") -> mujoco.MjSpec:
  """Load a fresh 29-DoF G1 spec.

  The source model uses primitive robot contacts and the supplied mesh
  colliders for the racket head, neck, and mount transition. Both accepted
  mode names therefore select the same validated contact model.
  """
  if collision_mode not in ("mesh", "primitive"):
    raise ValueError(f"Unknown collision mode: {collision_mode!r}")
  spec = mujoco.MjSpec.from_file(str(G1_PINGPONG_XML))
  # MjWarp reads qpos0 before the task's first reset. The fully extended
  # zero-angle legs require this height; the configured keyframe remains at
  # the original 0.76 m bent-knee pose.
  spec.body(G1_PINGPONG_ROOT_BODY).pos[:] = (0.0, 0.0, 0.82)
  return spec


def get_mesh_spec() -> mujoco.MjSpec:
  """Load the validated source contact model under its legacy mode name."""
  return get_spec("mesh")


def get_primitive_spec() -> mujoco.MjSpec:
  """Load the validated source contact model under its explicit mode name."""
  return get_spec("primitive")


def get_g1_pingpong_robot_cfg(
  collision_mode: CollisionMode = "mesh",
) -> EntityCfg:
  """Create the 29-DoF G1 entity with its validated native servo ordering."""
  if collision_mode not in ("mesh", "primitive"):
    raise ValueError(f"Unknown collision mode: {collision_mode!r}")

  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,),
      stiffness=settings["stiffness"],
      damping=settings["damping"],
      effort_limit=settings["effort_limit"],
      armature=settings["armature"],
    )
    for name, settings in MANIFEST["actuator_settings"].items()
  )
  return EntityCfg(
    spec_fn=get_mesh_spec if collision_mode == "mesh" else get_primitive_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.76),
      joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
      },
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=0.9,
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=0,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(1.0, 0.005, 0.0001),
      ),
    ),
  )


def get_robot_cfg(collision_mode: CollisionMode = "mesh") -> EntityCfg:
  """Source-compatible alias for :func:`get_g1_pingpong_robot_cfg`."""
  return get_g1_pingpong_robot_cfg(collision_mode)

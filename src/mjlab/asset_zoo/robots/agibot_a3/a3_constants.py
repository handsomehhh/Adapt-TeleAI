"""Agibot A3 ping-pong asset and position-servo configuration.

Adapted from HOPE's frozen August 2026 robot configuration; see ``README.md``
and ``LICENSE-HOPE`` in this directory for source and attribution. Motion files
carry joint/body names: their Isaac ordering must not be used as MuJoCo indices.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSET_DIR = Path(__file__).resolve().parent
A3_XML = ASSET_DIR / "xmls" / "a3_pingpong.xml"
_SETTINGS = json.loads((ASSET_DIR / "baseline_robot.json").read_text())
_MANIFEST = json.loads((ASSET_DIR / "source_manifest.json").read_text())

# Native MuJoCo kinematic-tree order, also the actuator/action order.
A3_JOINT_NAMES: tuple[str, ...] = tuple(_MANIFEST["primitive"]["joint_names"])
A3_BODY_NAMES: tuple[str, ...] = tuple(_MANIFEST["primitive"]["body_names"])
# Original Isaac ordering is exposed solely for legacy data conversion.
A3_ISAAC_JOINT_NAMES: tuple[str, ...] = tuple(_SETTINGS["joint_names"])
A3_ISAAC_BODY_NAMES: tuple[str, ...] = tuple(_SETTINGS["isaac_body_names"])

A3_ROOT_BODY = "pelvis_link"
A3_ANCHOR_BODY = "torso_Link"
A3_WRIST_BODY = "right_wrist_yaw_Link"
A3_FEET_BODIES = ("left_ankle_roll_Link", "right_ankle_roll_Link")
A3_HAND_BODIES = ("left_wrist_yaw_Link", "right_wrist_yaw_Link")
A3_RACKET_SITE = "racket_center"
A3_RACKET_COLLISION_GEOM = "pingpong_racket_collision"
A3_MOUNT_OFFSET = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
# The A3 paddle lies in the wrist's local XZ plane; its thin axis is local Y.
A3_RACKET_NORMAL = (0.0, 1.0, 0.0)
A3_TRACKED_BODIES = (
  "pelvis_link",
  "left_hip_roll_Link",
  "left_knee_Link",
  "left_ankle_roll_Link",
  "right_hip_roll_Link",
  "right_knee_Link",
  "right_ankle_roll_Link",
  "torso_Link",
  "left_shoulder_roll_Link",
  "left_elbow_Link",
  "left_wrist_yaw_Link",
  "right_shoulder_roll_Link",
  "right_elbow_Link",
  "right_wrist_yaw_Link",
)


def _resolve(value: dict[str, float] | float, joint_name: str) -> float:
  if not isinstance(value, dict):
    return float(value)
  matches = [v for pattern, v in value.items() if re.fullmatch(pattern, joint_name)]
  if len(matches) != 1:
    raise ValueError(f"Expected one actuator setting for {joint_name}: {value}")
  return float(matches[0])


def _joint_settings(joint_name: str) -> dict:
  matches = [
    group
    for group in _SETTINGS["robot"]["actuators"].values()
    if any(re.fullmatch(pattern, joint_name) for pattern in group["joint_names_expr"])
  ]
  if len(matches) != 1:
    raise ValueError(f"Expected one actuator group for {joint_name}")
  return matches[0]


def get_spec() -> mujoco.MjSpec:
  """Load a fresh A3 MJCF with primitive contacts and original visual meshes."""
  return mujoco.MjSpec.from_file(str(A3_XML))


def get_a3_robot_cfg() -> EntityCfg:
  """Create the A3 robot with the HOPE PD gains, armatures and torque limits.

  Self contacts are disabled, matching the reference motion-tracking setup.
  Ground/table contacts are enabled. Ball strikes are handled by the task's
  swept paddle-plane collision, avoiding discrete-step tunnelling.
  MuJoCo's implicit servos do not implement PhysX hard velocity clamps; the
  reference limits are exposed through ``A3_JOINT_VELOCITY_LIMITS``.
  """
  actuators = []
  for joint_name in A3_JOINT_NAMES:
    group = _joint_settings(joint_name)
    actuators.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=_resolve(group["stiffness"], joint_name),
        damping=_resolve(group["damping"], joint_name),
        effort_limit=_resolve(group["effort_limit_sim"], joint_name),
        armature=_resolve(group["armature"], joint_name),
      )
    )
  initial_state = deepcopy(_SETTINGS["robot"]["init_state"])
  for field in ("pos", "rot", "lin_vel", "ang_vel"):
    initial_state[field] = tuple(initial_state[field])
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(**initial_state),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(actuators),
      soft_joint_pos_limit_factor=_SETTINGS["robot"]["soft_joint_pos_limit_factor"],
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


A3_ACTION_SCALE: dict[str, float] = dict(_SETTINGS["action"]["scale"])
A3_JOINT_VELOCITY_LIMITS = {
  name: _resolve(_joint_settings(name)["velocity_limit_sim"], name)
  for name in A3_JOINT_NAMES
}

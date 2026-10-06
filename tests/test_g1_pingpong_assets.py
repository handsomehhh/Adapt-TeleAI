"""CPU checks for the 29-DoF G1 ping-pong asset contract."""

import hashlib

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  ASSET_DIR,
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_BODY_NAMES,
  G1_PINGPONG_JOINT_EFFORT_LIMITS,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_RACKET_NORMAL,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_SOURCE_URDF,
  G1_PINGPONG_WRIST_BODY,
  G1_PINGPONG_XML,
  MANIFEST,
  get_g1_pingpong_robot_cfg,
  get_spec,
)
from mjlab.entity import Entity


def _sha256(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def test_g1_pingpong_frozen_asset_checksums():
  assert _sha256(G1_PINGPONG_SOURCE_URDF) == MANIFEST["source"]["sha256"]
  assert _sha256(G1_PINGPONG_XML) == MANIFEST["mjcf_sha256"]
  assert {path.name for path in (ASSET_DIR / "meshes").iterdir()} == set(
    MANIFEST["meshes"]
  )
  for filename, source in MANIFEST["meshes"].items():
    assert _sha256(ASSET_DIR / "meshes" / filename) == source["sha256"]


def test_g1_pingpong_mjcf_kinematic_and_racket_contract():
  model = get_spec().compile()

  assert (model.nq, model.nv, model.nu) == (36, 35, 0)
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == (
    G1_PINGPONG_JOINT_NAMES
  )
  assert tuple(model.body(i).name for i in range(1, model.nbody)) == (
    G1_PINGPONG_BODY_NAMES
  )
  assert len(G1_PINGPONG_JOINT_NAMES) == 29
  assert len(G1_PINGPONG_BODY_NAMES) == 30
  np.testing.assert_allclose(model.body_mass.sum(), 33.68916754973783, rtol=1e-12)
  assert model.qpos0[2] == 0.82

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  wrist = model.body(G1_PINGPONG_WRIST_BODY)
  site = model.site(G1_PINGPONG_RACKET_SITE)
  assert site.bodyid == wrist.id
  np.testing.assert_allclose(site.pos, G1_PINGPONG_MOUNT_OFFSET, atol=1e-14)
  wrist_rotation = data.xmat[wrist.id].reshape(3, 3)
  np.testing.assert_allclose(
    data.site_xmat[site.id].reshape(3, 3)[:, 1],
    wrist_rotation @ np.asarray(G1_PINGPONG_RACKET_NORMAL),
    atol=1e-12,
  )


def test_g1_pingpong_entity_actuators_and_initial_pose():
  model = Entity(get_g1_pingpong_robot_cfg()).spec.compile()

  assert (model.nq, model.nv, model.nu) == (36, 35, 29)
  np.testing.assert_array_equal(model.actuator_trnid[:, 0], range(1, 30))
  np.testing.assert_allclose(model.actuator_forcerange, model.jnt_actfrcrange[1:])
  assert np.all(model.actuator_forcelimited)
  assert np.all(model.dof_armature[6:] > 0.0)

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  assert data.qpos[2] == 0.76
  for name in G1_PINGPONG_JOINT_NAMES:
    expected = 0.0
    if "hip_pitch_joint" in name:
      expected = -0.312
    elif "knee_joint" in name:
      expected = 0.669
    elif "ankle_pitch_joint" in name:
      expected = -0.363
    elif "elbow_joint" in name:
      expected = 0.6
    elif name in ("left_shoulder_roll_joint", "left_shoulder_pitch_joint"):
      expected = 0.2
    elif name == "right_shoulder_roll_joint":
      expected = -0.2
    elif name == "right_shoulder_pitch_joint":
      expected = 0.2
    assert data.joint(name).qpos[0] == expected


def test_g1_pingpong_action_scales_follow_servo_contract():
  assert tuple(G1_PINGPONG_ACTION_SCALE) == G1_PINGPONG_JOINT_NAMES
  assert tuple(G1_PINGPONG_JOINT_EFFORT_LIMITS) == G1_PINGPONG_JOINT_NAMES

  for name in G1_PINGPONG_JOINT_NAMES:
    settings = MANIFEST["actuator_settings"][name]
    expected = 0.25 * settings["effort_limit"] / settings["stiffness"]
    assert G1_PINGPONG_ACTION_SCALE[name] == expected
    assert G1_PINGPONG_JOINT_EFFORT_LIMITS[name] == settings["effort_limit"]

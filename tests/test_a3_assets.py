"""CPU checks for the A3 model's motion/actuation contract."""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.agibot_a3.a3_constants import (
  A3_BODY_NAMES,
  A3_JOINT_NAMES,
  A3_MOUNT_OFFSET,
  A3_RACKET_COLLISION_GEOM,
  A3_RACKET_NORMAL,
  A3_RACKET_SITE,
  A3_WRIST_BODY,
  get_a3_robot_cfg,
)
from mjlab.entity import Entity


def test_a3_compiled_motion_and_actuator_contract():
  robot = Entity(get_a3_robot_cfg())
  model = robot.spec.compile()
  assert (model.nq, model.nv, model.nu) == (38, 37, 31)
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == A3_JOINT_NAMES
  assert tuple(model.body(i).name for i in range(1, model.nbody)) == A3_BODY_NAMES
  assert len(A3_BODY_NAMES) == 32
  np.testing.assert_allclose(model.body_mass.sum(), 58.27723163, rtol=1e-9)
  assert np.all(model.dof_armature[6:] > 0.0)

  # Each native servo must address the corresponding joint and enforce the
  # original joint torque limit, including the 6 Nm wrists and 320 Nm knees.
  np.testing.assert_array_equal(model.actuator_trnid[:, 0], range(1, 32))
  np.testing.assert_allclose(model.actuator_forcerange, model.jnt_actfrcrange[1:])
  assert np.all(model.actuator_forcelimited)


def test_a3_paddle_site_and_initial_ground_clearance():
  robot = Entity(get_a3_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  wrist = model.body(A3_WRIST_BODY)
  site = model.site(A3_RACKET_SITE)
  assert site.bodyid == wrist.id
  np.testing.assert_allclose(site.pos, A3_MOUNT_OFFSET, atol=1e-14)
  wrist_rotation = data.xmat[wrist.id].reshape(3, 3)
  np.testing.assert_allclose(
    data.site_xpos[site.id],
    data.xpos[wrist.id] + wrist_rotation @ np.asarray(A3_MOUNT_OFFSET),
    atol=1e-12,
  )
  np.testing.assert_allclose(
    data.site_xmat[site.id].reshape(3, 3)[:, 1],
    wrist_rotation @ np.asarray(A3_RACKET_NORMAL),
    atol=1e-12,
  )
  paddle = model.geom(A3_RACKET_COLLISION_GEOM)
  assert np.argmin(paddle.size) == 1  # The thin paddle axis is local Y.

  # A newly compiled model must not start embedded in the ground before the
  # first motion reset. Also verify the ready-pose feet are near floor height.
  assert model.qpos0[2] == 1.0684
  for side in ("left", "right"):
    foot = model.geom(f"{side}_ankle_roll_Link_0_collision")
    rotation = data.geom_xmat[foot.id].reshape(3, 3)
    bottom = data.geom_xpos[foot.id, 2] - np.abs(rotation[2]) @ foot.size
    assert -0.01 < bottom < 0.03
    assert foot.contype == 0
    assert foot.conaffinity == 1

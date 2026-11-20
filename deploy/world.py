"""Native CPU MuJoCo plant constructed with the training scene and servo config."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_BODY_NAMES,
  G1_PINGPONG_JOINT_NAMES,
)
from mjlab.scene import Scene
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg
from mjlab.tasks.g1_adapt.runtime import RobotState
from mjlab.utils.lab_api.math import quat_apply_inverse


class World:
  def __init__(self):
    cfg = g1_adapt_env_cfg(play=True)
    cfg.scene.num_envs = 1
    self.model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(self.model)
    self.data = mujoco.MjData(self.model)
    self.dt = cfg.sim.mujoco.timestep
    self.control_dt = self.dt * cfg.decimation
    self.body_names = tuple(G1_PINGPONG_BODY_NAMES)
    self.body_ids = [self.model.body("robot/" + name).id for name in self.body_names]
    joint_ids = [
      self.model.joint("robot/" + name).id for name in G1_PINGPONG_JOINT_NAMES
    ]
    self.qids = self.model.jnt_qposadr[joint_ids]
    self.vids = self.model.jnt_dofadr[joint_ids]
    self.actuators = [
      int(np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)[0])
      for jid in joint_ids
    ]
    self.default_q = torch.tensor(
      self.model.key("init_state").qpos[self.qids].copy(), dtype=torch.float32
    )
    self.proxy = SimpleNamespace(body_names=self.body_names, data=SimpleNamespace())
    self.refresh()

  def reset(self, reference):
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:3] = reference.root[0].numpy()
    self.data.qpos[3:7] = reference.quaternions[0, 0].numpy()
    self.data.qpos[self.qids] = reference.state[0, 12:41].numpy()
    self.data.qvel[:3] = reference.state[0, :3].numpy()
    self.data.qvel[3:6] = quat_apply_inverse(
      reference.quaternions[:, 0], reference.state[:, 3:6]
    )[0].numpy()
    self.data.qvel[self.vids] = reference.state[0, 41:70].numpy()
    self.data.ctrl[self.actuators] = self.data.qpos[self.qids]
    self.refresh()

  def refresh(self, forward=True):
    if forward:
      mujoco.mj_forward(self.model, self.data)
    p = torch.tensor(self.data.xpos[self.body_ids].copy(), dtype=torch.float32)[None]
    q = torch.tensor(self.data.xquat[self.body_ids].copy(), dtype=torch.float32)[None]
    # mj_objectVelocity(BODY, local=1) uses the body's inertial orientation.
    # The trained observation instead rotates world cvel into the link frame.
    angular = torch.tensor(
      self.data.cvel[self.body_ids[0], :3].copy(), dtype=torch.float32
    )[None]
    self.robot_state = RobotState(
      p[:, 0],
      q[:, 0],
      torch.tensor(self.data.qpos[self.qids].copy(), dtype=torch.float32)[None],
      torch.tensor(self.data.qvel[self.vids].copy(), dtype=torch.float32)[None],
      quat_apply_inverse(q[:, 0], angular),
    )
    gravity = quat_apply_inverse(q[:, 0], torch.tensor([[0.0, 0, -1.0]]))
    self.proxy.data = SimpleNamespace(
      body_link_pos_w=p,
      body_link_quat_w=q,
      root_link_pos_w=p[:, 0],
      projected_gravity_b=gravity,
    )

  def step(self, target):
    values = np.asarray(target).reshape(29)
    if not np.isfinite(values).all():
      raise ValueError("Non-finite joint targets")
    self.data.ctrl[self.actuators] = values
    mujoco.mj_step(self.model, self.data)
    # Training reads pre-integration derived poses for substep ball contacts,
    # then calls forward once at the end of the whole control period.
    self.refresh(forward=False)

  @property
  def fallen(self):
    return bool(
      self.proxy.data.root_link_pos_w[0, 2] < 0.45
      or self.proxy.data.projected_gravity_b[0, 2] > -0.5
    )

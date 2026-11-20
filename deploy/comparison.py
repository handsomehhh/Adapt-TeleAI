"""A complete reference robot drawn from NPZ joints, independent of the plant."""

import copy

import mujoco
import numpy as np
from mjviser.conversions import merge_geoms


class ReferenceRobot:
  def __init__(self, scene, world):
    self.model = copy.copy(world.model)
    self.data = mujoco.MjData(self.model)
    self.qids = world.qids
    self.frame = scene.add_frame("/reference_robot", show_axes=False)
    self.handles = {}
    for body_id in range(1, self.model.nbody):
      name = self.model.body(body_id).name
      if not name or not name.startswith("robot/"):
        continue
      geoms = [
        i
        for i in range(self.model.ngeom)
        if self.model.geom_bodyid[i] == body_id
        and self.model.geom_group[i] == 2
        and self.model.geom_rgba[i, 3] > 0
      ]
      if not geoms:
        continue
      mesh = merge_geoms(self.model, geoms)
      self.handles[body_id] = scene.add_mesh_simple(
        f"/reference_robot/{name}",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(45, 205, 110),
        opacity=0.65,
        cast_shadow=False,
        receive_shadow=False,
      )
    self.label = scene.add_label("/reference_robot/label", "参考 motion")
    self.opacity = 0.65

  def update(self, qpos, offset, visible=True, opacity=0.65):
    # Rendering has its own data and never executes mj_step or writes plant state.
    self.data.qpos[:] = qpos
    mujoco.mj_kinematics(self.model, self.data)
    self.frame.position = np.asarray(offset)
    self.frame.visible = visible
    for body_id, handle in self.handles.items():
      handle.position = self.data.xpos[body_id].copy()
      handle.wxyz = self.data.xquat[body_id].copy()
      if opacity != self.opacity:
        handle.opacity = opacity
    self.label.position = self.data.qpos[:3] + np.array([0.0, 0, 0.9])
    self.opacity = opacity


def reference_qpos(world, reference):
  """Map canonical reference joints into the actual model's qpos addresses."""
  qpos = world.data.qpos.copy()
  qpos[:3] = reference.root[0].numpy()
  qpos[3:7] = reference.quaternions[0, 0].numpy()
  qpos[world.qids] = reference.state[0, 12:41].numpy()
  return qpos

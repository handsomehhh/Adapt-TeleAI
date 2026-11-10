"""Position-only target ghosts matching PingPong's 1v0/1v1 target overlays."""

import numpy as np


def command_geometry(model):
  """Record stable model names, so replay needs no robot assets or planner."""
  import mujoco

  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
  site = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "robot/right_racket_surface"
  )
  if pelvis < 0 or site < 0:
    return None
  base, racket = [], []
  for i in range(model.ngeom):
    name = model.geom(i).name
    if (
      model.geom_bodyid[i] == pelvis
      and model.geom_contype[i] == 0
      and model.geom_conaffinity[i] == 0
      and model.geom_rgba[i, 3] > 0
    ):
      base.append(name)
    if model.geom_bodyid[i] == model.site_bodyid[site] and name.rsplit("/", 1)[-1] in (
      "right_racket_link_1_visual",
      "right_racket_link_2_visual",
      "right_racket_link_0_collision",
      "right_racket_handle_collision",
    ):
      racket.append(name)
  if not base or not racket:
    return None
  return dict(
    pelvis_body=model.body(pelvis).name,
    racket_site=model.site(site).name,
    pelvis_geoms=base,
    racket_geoms=racket,
    source="controller state goal_base_xy and goal_position in hope_table",
    orientation="measured geometry orientation; pelvis Z stays at measured height",
  )


class CommandGhosts:
  """Translate copies of measured geometry without changing the measured qpos."""

  def __init__(self, server, model, definition):
    import mujoco
    from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

    self.handles = {"pelvis": {}, "racket": {}}
    self.available = definition is not None
    if not self.available:
      return
    self.pelvis = model.body(definition["pelvis_body"]).id
    self.site = model.site(definition["racket_site"]).id
    for group, color, opacity in (
      ("pelvis", (38, 255, 115), 0.35),
      ("racket", (255, 191, 26), 0.45),
    ):
      for name in definition[group + "_geoms"]:
        geom = model.geom(name).id
        mesh = (
          mujoco_mesh_to_trimesh(model, geom)
          if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
          else create_primitive_mesh(model, geom)
        )
        self.handles[group][geom] = server.scene.add_mesh_simple(
          f"/goals/{group}/{geom}",
          mesh.vertices,
          mesh.faces,
          color=color,
          opacity=opacity,
          cast_shadow=False,
          receive_shadow=False,
          visible=False,
        )

  def update(self, data, row, offset, *, visible):
    from viser.transforms import SO3

    if self.available and visible:
      base_target = np.r_[
        np.asarray(row["goal_base_xy"]) - offset[:2], data.xpos[self.pelvis, 2]
      ]
      deltas = dict(
        pelvis=base_target - data.xpos[self.pelvis],
        racket=np.asarray(row["goal_position"]) - offset - data.site_xpos[self.site],
      )
    for group, handles in self.handles.items():
      for geom, handle in handles.items():
        handle.visible = bool(visible)
        if visible:
          handle.position = data.geom_xpos[geom] + deltas[group]
          handle.wxyz = SO3.from_matrix(data.geom_xmat[geom].reshape(3, 3)).wxyz

"""HOPE court visualization aligned with the analytical ball surfaces."""

import mujoco
import numpy as np

from mjlab.entity import EntityCfg


def table_cfg(ball):
  near, far, width, height, net_height = (
    float(ball[name])
    for name in (
      "table_near_x",
      "table_far_x",
      "table_width",
      "table_height",
      "net_height",
    )
  )
  if (
    not np.isfinite([near, far, width, height, net_height]).all()
    or far <= near
    or min(width, height, net_height) <= 0
  ):
    raise ValueError("Invalid HOPE table dimensions")
  center = (near + far) / 2
  half_length, half_width = (far - near) / 2, width / 2
  net_overhang = float(ball.get("net_overhang", 0))
  if not np.isfinite(net_overhang) or net_overhang < 0:
    raise ValueError("Invalid HOPE net overhang")

  def spec_fn():
    spec = mujoco.MjSpec()
    table = spec.worldbody.add_body(name="table")

    def box(name, pos, size, rgba):
      # Ball/table contacts are integrated analytically by SimBall.
      table.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=rgba,
        contype=0,
        conaffinity=0,
      )

    box(
      "top",
      (center, 0, height - 0.015),
      (half_length, half_width, 0.015),
      (0.05, 0.25, 0.42, 1),
    )
    box(
      "net",
      (center, 0, height + net_height / 2),
      (0.003, half_width + net_overhang, net_height / 2),
      (0.7, 0.7, 0.7, 0.75),
    )
    line_z = height + 0.0005
    for name, x in (("near_line", near + 0.01), ("far_line", far - 0.01)):
      box(name, (x, 0, line_z), (0.01, half_width, 0.0005), (1, 1, 1, 1))
    for name, y in (
      ("left_line", -half_width + 0.01),
      ("right_line", half_width - 0.01),
    ):
      box(name, (center, y, line_z), (half_length, 0.01, 0.0005), (1, 1, 1, 1))
    box("center_line", (center, 0, line_z), (half_length, 0.0015, 0.0005), (1, 1, 1, 1))
    return spec

  return EntityCfg(spec_fn=spec_fn)

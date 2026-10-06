"""Regulation-size table geometry, matching the analytical ball surfaces."""

import mujoco

from mjlab.entity import EntityCfg


def table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  table = spec.worldbody.add_body(name="table")
  table.add_geom(
    name="top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(2.02, 0, 0.745),
    size=(1.37, 0.7625, 0.015),
    rgba=(0.05, 0.25, 0.42, 1),
    contype=1,
    conaffinity=1,
  )
  table.add_geom(
    name="net",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(2.02, 0, 0.83625),
    size=(0.003, 0.7625, 0.07625),
    rgba=(0.7, 0.7, 0.7, 0.75),
    contype=1,
    conaffinity=1,
  )
  return spec


def table_cfg():
  return EntityCfg(spec_fn=table_spec)

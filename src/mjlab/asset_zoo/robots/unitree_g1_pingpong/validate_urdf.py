"""Compare the training model against MuJoCo's independent native URDF import."""

import argparse
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mjlab.entity import Entity

from .convert_urdf import digest, parallel_axis, pose
from .g1_constants import (
  ASSET_DIR,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_SOURCE_URDF,
  G1_PINGPONG_XML,
  MANIFEST,
  get_g1_pingpong_robot_cfg,
)


def validate(samples=32):
  source = ET.parse(G1_PINGPONG_SOURCE_URDF).getroot()
  compiler = source.find("mujoco/compiler")
  assert compiler is not None
  compiler.set("meshdir", str(ASSET_DIR / "meshes"))
  compiler.set("fusestatic", "false")
  compiler.set("discardvisual", "false")
  reference = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
  # The native compiler diagonalizes full inertias approximately. Preserve the
  # original URDF tensors for an independent physical-inertia comparison.
  source_inertias = {}
  for link in source.findall("link"):
    inertial = link.find("inertial")
    if inertial is None:
      source_inertias[link.attrib["name"]] = np.zeros((3, 3))
      continue
    inertia = inertial.find("inertia")
    assert inertia is not None
    xx, yy, zz, xy, xz, yz = [
      float(inertia.attrib[key]) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    ]
    rotation = pose(inertial.find("origin"))[:3, :3]
    source_inertias[link.get("name")] = (
      rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
    )
  model = mujoco.MjModel.from_xml_path(str(G1_PINGPONG_XML))
  ref_data, data = mujoco.MjData(reference), mujoco.MjData(model)
  groups = {}
  for body_id in range(1, reference.nbody):
    ancestor = body_id
    while (
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, reference.body(ancestor).name)
      < 0
    ):
      ancestor = int(reference.body_parentid[ancestor])
    groups.setdefault(reference.body(ancestor).name, []).append(body_id)
  maximum = {
    name: 0.0
    for name in (
      "fk_position",
      "fk_rotation",
      "com",
      "inertia",
      "geom_position",
      "geom_rotation",
      "racket_position",
      "racket_rotation",
    )
  }
  rng = np.random.default_rng(42)
  for _ in range(samples):
    # Native URDF has a fixed root at the origin. Match it exactly here.
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for name in MANIFEST["mesh"]["joint_names"]:
      joint = model.joint(name)
      q = rng.uniform(*joint.range)
      data.qpos[joint.qposadr[0]] = q
      ref_data.qpos[reference.joint(name).qposadr[0]] = q
      np.testing.assert_array_equal(joint.range, reference.joint(name).range)
    mujoco.mj_forward(reference, ref_data)
    mujoco.mj_forward(model, data)
    for name, members in groups.items():
      actual, expected = model.body(name).id, reference.body(name).id
      maximum["fk_position"] = max(
        maximum["fk_position"],
        float(np.max(np.abs(data.xpos[actual] - ref_data.xpos[expected]))),
      )
      maximum["fk_rotation"] = max(
        maximum["fk_rotation"],
        float(np.max(np.abs(data.xmat[actual] - ref_data.xmat[expected]))),
      )
      mass = reference.body_mass[members].sum()
      center = (reference.body_mass[members, None] * ref_data.xipos[members]).sum(
        0
      ) / mass
      inertia = np.zeros((3, 3))
      for member in members:
        rotation = ref_data.xmat[member].reshape(3, 3)
        inertia += rotation @ source_inertias[
          reference.body(member).name
        ] @ rotation.T + reference.body_mass[member] * parallel_axis(
          ref_data.xipos[member] - center
        )
      rotation = data.ximat[actual].reshape(3, 3)
      actual_inertia = rotation @ np.diag(model.body_inertia[actual]) @ rotation.T
      np.testing.assert_allclose(model.body_mass[actual], mass, atol=1e-12)
      maximum["com"] = max(
        maximum["com"], float(np.max(np.abs(data.xipos[actual] - center)))
      )
      maximum["inertia"] = max(
        maximum["inertia"], float(np.max(np.abs(actual_inertia - inertia)))
      )
    # Compare every source visual/collision at its full world transform. Native
    # URDF import retains the fixed links; training folds their geoms into owners.
    for link in source.findall("link"):
      link_name = link.attrib["name"]
      ref_body = reference.body(link_name).id
      ref_geoms = list(
        range(
          int(reference.body_geomadr[ref_body]),
          int(reference.body_geomadr[ref_body] + reference.body_geomnum[ref_body]),
        )
      )
      for kind in ("visual", "collision"):
        expected_ids = [
          index
          for index in ref_geoms
          if bool(reference.geom_contype[index] or reference.geom_conaffinity[index])
          == (kind == "collision")
        ]
        assert len(expected_ids) == len(link.findall(kind))
        for index, expected in enumerate(expected_ids):
          actual = model.geom(f"{link_name}_{index}_{kind}").id
          assert model.geom_type[actual] == reference.geom_type[expected]
          np.testing.assert_allclose(
            model.geom_size[actual], reference.geom_size[expected], atol=1e-12
          )
          maximum["geom_position"] = max(
            maximum["geom_position"],
            float(
              np.max(np.abs(data.geom_xpos[actual] - ref_data.geom_xpos[expected]))
            ),
          )
          maximum["geom_rotation"] = max(
            maximum["geom_rotation"],
            float(
              np.max(np.abs(data.geom_xmat[actual] - ref_data.geom_xmat[expected]))
            ),
          )
    site = model.site(G1_PINGPONG_RACKET_SITE).id
    expected = reference.body("right_racket_link").id
    maximum["racket_position"] = max(
      maximum["racket_position"],
      float(np.max(np.abs(data.site_xpos[site] - ref_data.xpos[expected]))),
    )
    maximum["racket_rotation"] = max(
      maximum["racket_rotation"],
      float(np.max(np.abs(data.site_xmat[site] - ref_data.xmat[expected]))),
    )
  # Native URDF and MJCF importers differ slightly when principalizing meshes.
  # This tolerance is below 0.0003 degrees; all translational and physical
  # inertia comparisons retain the stricter numerical tolerance.
  tolerances = {name: (5e-6 if name == "geom_rotation" else 1e-9) for name in maximum}
  for name, error in maximum.items():
    if error > tolerances[name]:
      raise ValueError(f"URDF/MJCF validation failed: {name} error={error}")
  np.testing.assert_allclose(
    model.body_mass.sum(), reference.body_mass.sum(), atol=1e-12
  )
  entity = Entity(get_g1_pingpong_robot_cfg()).spec.compile()
  np.testing.assert_array_equal(entity.actuator_trnid[:, 0], np.arange(1, 30))
  np.testing.assert_allclose(entity.actuator_forcerange, entity.jnt_actfrcrange[1:])
  report = {
    "asset": {
      "samples": samples,
      "reference": "MuJoCo native URDF FK and geoms, fusestatic=false; original URDF full inertia tensors with all fixed-link masses retained",
      "dofs": model.njnt - 1,
      "articulated_bodies": model.nbody - 1,
      "meshes": model.nmesh,
      "geoms": model.ngeom,
      "collision_geoms": int(
        np.count_nonzero(model.geom_contype | model.geom_conaffinity)
      ),
      "source_mass_kg": float(reference.body_mass.sum()),
      "compiled_mass_kg": float(model.body_mass.sum()),
      "maximum_errors": maximum,
      "error_tolerances": tolerances,
      "limits_exact": True,
      "self_collision_disabled": bool(np.all(model.geom_contype == 0)),
    },
    "entity": {
      "entity_compiles": True,
      "nq": entity.nq,
      "nv": entity.nv,
      "nu": entity.nu,
      "action_dimension": entity.nu,
      "servo_order_and_effort_limits_exact": True,
    },
    "source_urdf_sha256": digest(G1_PINGPONG_SOURCE_URDF),
    "mjcf_sha256": digest(G1_PINGPONG_XML),
    "robot_module_sha256": digest(ASSET_DIR / "g1_constants.py"),
  }
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--samples", type=int, default=32)
  parser.add_argument("--write-report", action="store_true")
  args = parser.parse_args()
  report = validate(args.samples)
  if args.write_report:
    (ASSET_DIR / "asset_validation.json").write_text(
      json.dumps(report, indent=2) + "\n"
    )
    manifest = json.loads((ASSET_DIR / "asset_manifest.json").read_text())
    manifest["validation"] = report["asset"]
    (ASSET_DIR / "asset_manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n"
    )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

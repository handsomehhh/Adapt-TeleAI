"""Update the G1 training MJCF while preserving its 29-joint servo ordering.

Fixed-link geometry and full inertias are merged into their articulated owner.
The existing training collision masks and simulation defaults are preserved.
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent


def required(parent, path):
  node = parent.find(path)
  if node is None:
    raise ValueError(f"Missing URDF element: {path}")
  return node


def numbers(value):
  return np.fromstring(value, sep=" ")


def formatted(value):
  return " ".join(f"{float(v):.17g}" for v in np.asarray(value).reshape(-1))


def pose(origin):
  transform = np.eye(4)
  if origin is not None:
    transform[:3, 3] = numbers(origin.get("xyz", "0 0 0"))
    roll, pitch, yaw = numbers(origin.get("rpy", "0 0 0"))
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    transform[:3, :3] = [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ]
  return transform


def quaternion(rotation):
  result = np.zeros(4)
  mujoco.mju_mat2Quat(result, np.ascontiguousarray(rotation).reshape(-1))
  return result


def parallel_axis(position):
  return np.dot(position, position) * np.eye(3) - np.outer(position, position)


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_asset(urdf_path, mesh_directory=None):
  source = Path(urdf_path).resolve()
  source_root = ET.parse(source).getroot()
  source_meshes = (
    Path(mesh_directory).resolve()
    if mesh_directory
    else source.parent.parent / "meshes"
  )
  manifest_path = ASSET_DIR / "asset_manifest.json"
  manifest = json.loads(manifest_path.read_text())
  tree = ET.parse(ASSET_DIR / "g1.xml")
  root = tree.getroot()
  bodies = {node.get("name"): node for node in root.findall(".//body")}
  joints = {node.get("name"): node for node in source_root.findall("joint")}
  moving = [node for node in joints.values() if node.get("type") != "fixed"]
  if {node.get("name") for node in moving} != set(manifest["mesh"]["joint_names"]):
    raise ValueError("URDF must retain the native 29-joint articulation")
  # This updater supports geometry/inertia revisions without silently changing
  # the joint transforms expected by the independent motion archive.
  current = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "g1.xml"))
  for joint in moving:
    name = joint.attrib["name"]
    child = required(joint, "child").get("link")
    expected = pose(joint.find("origin"))
    body = current.body(child)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, body.quat)
    if not (
      np.allclose(body.pos, expected[:3, 3], atol=1e-12)
      and np.allclose(rotation.reshape(3, 3), expected[:3, :3], atol=1e-10)
      and np.allclose(
        current.joint(name).axis,
        numbers(required(joint, "axis").get("xyz")),
        atol=1e-12,
      )
    ):
      raise ValueError(f"Joint transform changed; retarget motion data first: {name}")
    limit = required(joint, "limit")
    if not np.array_equal(
      current.joint(name).range, [float(limit.get("lower")), float(limit.get("upper"))]
    ):
      raise ValueError(f"Joint range changed: {name}")
  for body in bodies.values():
    for node in list(body):
      if node.tag in ("geom", "inertial", "site"):
        body.remove(node)
  asset = required(root, "asset")
  asset.clear()
  links = {node.get("name"): node for node in source_root.findall("link")}
  children = {}
  for joint in joints.values():
    children.setdefault(required(joint, "parent").get("link"), []).append(joint)
  components = {name: [] for name in bodies}
  meshes = {}
  fixed_frames = {}
  collision_geoms = {name: [] for name in bodies}

  def visit(link_name, owner, transform):
    link = links[link_name]
    body = bodies[owner]
    inertial = link.find("inertial")
    if inertial is not None:
      mass = float(required(inertial, "mass").get("value"))
      frame = transform @ pose(inertial.find("origin"))
      values = required(inertial, "inertia").attrib
      xx, yy, zz, xy, xz, yz = [
        float(values[key]) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
      ]
      inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
      inertia = frame[:3, :3] @ inertia @ frame[:3, :3].T
      components[owner].append((mass, frame[:3, 3], inertia))
    if link_name != owner:
      fixed_frames[link_name] = {
        "body": owner,
        "pos": transform[:3, 3].tolist(),
        "quat_wxyz": quaternion(transform[:3, :3]).tolist(),
      }
    for kind in ("visual", "collision"):
      for index, element in enumerate(link.findall(kind)):
        shape = list(required(element, "geometry"))[0]
        frame = transform @ pose(element.find("origin"))
        name = f"{link_name}_{index}_{kind}"
        geom = ET.SubElement(
          body,
          "geom",
          {
            "name": name,
            "class": kind,
            "type": shape.tag,
            "pos": formatted(frame[:3, 3]),
            "quat": formatted(quaternion(frame[:3, :3])),
          },
        )
        if shape.tag == "mesh":
          filename = Path(shape.get("filename")).name
          if shape.get("scale") and not np.array_equal(
            numbers(shape.get("scale")), np.ones(3)
          ):
            raise ValueError("Scaled meshes require a distinct compiled mesh asset")
          meshes[filename] = source_meshes / filename
          geom.set("mesh", Path(filename).stem)
        elif shape.tag == "box":
          geom.set("size", formatted(numbers(shape.get("size")) / 2))
        elif shape.tag == "sphere":
          geom.set("size", shape.get("radius"))
        elif shape.tag == "cylinder":
          geom.set(
            "size",
            formatted([float(shape.get("radius")), float(shape.get("length")) / 2]),
          )
        else:
          raise ValueError(f"Unsupported URDF shape: {shape.tag}")
        if kind == "visual":
          color = element.find("material/color")
          geom.set(
            "rgba", color.attrib["rgba"] if color is not None else "0.7 0.7 0.7 1"
          )
        else:
          collision_geoms[owner].append(name)
    for joint in children.get(link_name, []):
      child = required(joint, "child").get("link")
      if joint.get("type") == "fixed":
        visit(child, owner, transform @ pose(joint.find("origin")))
      else:
        visit(child, child, np.eye(4))

  visit("pelvis", "pelvis", np.eye(4))
  for name, values in components.items():
    mass = sum(value[0] for value in values)
    center = sum(value[0] * value[1] for value in values) / mass
    inertia = sum(
      value[2] + value[0] * parallel_axis(value[1]) for value in values
    ) - mass * parallel_axis(center)
    principal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
      axes[:, 0] *= -1
    node = ET.Element(
      "inertial",
      mass=formatted([mass]),
      pos=formatted(center),
      diaginertia=formatted(principal),
      quat=formatted(quaternion(axes)),
    )
    bodies[name].insert(0, node)
  for filename in meshes:
    ET.SubElement(asset, "mesh", name=Path(filename).stem, file=filename)
  racket = fixed_frames["right_racket_link"]
  ET.SubElement(
    bodies[racket["body"]],
    "site",
    name="right_racket_surface",
    pos=formatted(racket["pos"]),
    quat=formatted(racket["quat_wxyz"]),
    size="0.003",
    group="5",
    rgba="1 0.1 0.1 1",
  )
  root.set("model", source.stem)
  ET.indent(root, space="  ")
  xml = ET.tostring(root, encoding="unicode") + "\n"
  compiled = mujoco.MjModel.from_xml_string(
    xml, assets={name: path.read_bytes() for name, path in meshes.items()}
  )
  if tuple(compiled.joint(i).name for i in range(1, compiled.njnt)) != tuple(
    manifest["mesh"]["joint_names"]
  ):
    raise ValueError("Conversion changed the native joint order")
  for filename, path in meshes.items():
    destination = ASSET_DIR / "meshes" / filename
    if path.resolve() != destination.resolve():
      shutil.copy2(path, destination)
  for path in (ASSET_DIR / "meshes").iterdir():
    if path.name in manifest["meshes"] and path.name not in meshes:
      path.unlink()
  destination = ASSET_DIR / "source" / source.name
  if source != destination:
    shutil.copy2(source, destination)
  (ASSET_DIR / "g1.xml").write_text(xml)
  manifest["source"] = {
    "path": str(source),
    "local_path": f"source/{source.name}",
    "sha256": digest(source),
  }
  manifest["meshes"] = {
    name: {"source": str(path), "sha256": digest(path)} for name, path in meshes.items()
  }
  manifest["fixed_frames"] = fixed_frames
  manifest["racket"] = {**manifest["racket"], **racket}
  manifest["mesh"]["collision_geoms"] = collision_geoms
  manifest["mesh"]["total_mass"] = float(compiled.body_mass.sum())
  manifest["conversion"] = (
    "analytic fixed-link inertia merge; supplied fast extended-27mm URDF shapes; training self-collision masks preserved"
  )
  manifest["mjcf_sha256"] = digest(ASSET_DIR / "g1.xml")
  manifest["validation"] = {"status": "pending independent URDF/MJCF validation"}
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  return {
    "source_sha256": manifest["source"]["sha256"],
    "mass_kg": manifest["mesh"]["total_mass"],
    "joints": compiled.njnt - 1,
    "bodies": compiled.nbody - 1,
    "geoms": compiled.ngeom,
    "collision_geoms": sum(map(len, collision_geoms.values())),
    "racket": manifest["racket"],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--urdf", type=Path, required=True)
  parser.add_argument("--mesh-directory", type=Path)
  args = parser.parse_args()
  print(json.dumps(update_asset(args.urdf, args.mesh_directory), indent=2))


if __name__ == "__main__":
  main()

"""Use the G1 checkpoint's own fast-URDF collision shapes in the local model."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import quaternion_multiply, rotation


def urdf_pose(origin):
  if origin is None:
    return np.zeros(3), np.array([1.0, 0, 0, 0])
  position = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
  roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
  quaternion = quaternion_multiply(
    quaternion_multiply(
      [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
      [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0],
    ),
    [np.cos(roll / 2), np.sin(roll / 2), 0, 0],
  )
  return position, quaternion


def training_collision_spec(base_factory, urdf, self_collisions):
  """Preserve the existing joint/inertial/visual asset; replace collision only.

  Fixed URDF links (head, mount, paddle) are expressed in their nearest MuJoCo
  ancestor body. No mass is inferred from the newly added collision geometry.
  """
  spec = base_factory()
  root = ET.parse(urdf).getroot()
  parents = {j.find("child").get("link"): j for j in root.findall("joint")}
  bodies = {body.name: body for body in spec.bodies}
  meshes = {mesh.name for mesh in spec.meshes}
  for geom in list(spec.geoms):
    if geom.name.endswith("_collision"):
      spec.delete(geom)
  count = 0
  for link in root.findall("link"):
    for collision in link.findall("collision"):
      body = link.get("name")
      pos, quat = urdf_pose(collision.find("origin"))
      while body not in bodies:
        joint = parents[body]
        p, q = urdf_pose(joint.find("origin"))
        pos, quat = p + rotation(q) @ pos, quaternion_multiply(q, quat)
        body = joint.find("parent").get("link")
      geometry = list(collision.find("geometry"))[0]
      kind = geometry.tag
      args = {}
      if kind == "box":
        args.update(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=np.fromstring(geometry.get("size"), sep=" ") / 2,
        )
      elif kind == "sphere":
        args.update(
          type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(geometry.get("radius")), 0, 0]
        )
      elif kind == "mesh":
        name = Path(geometry.get("filename")).stem
        if name not in meshes:
          raise ValueError(f"Training collision mesh is unavailable locally: {name}")
        if geometry.get("scale", "1 1 1") != "1 1 1":
          raise ValueError("Unsupported non-unit collision mesh scale")
        args.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
      else:
        raise ValueError(f"Unsupported training collision geometry: {kind}")
      bodies[body].add_geom(
        name=f"hope_training_shape_{count}",
        pos=pos,
        quat=quat,
        contype=1 if self_collisions else 0,
        conaffinity=1,
        condim=3,
        friction=[1.0, 0.005, 0.0001],
        mass=0.0,
        group=3,
        rgba=[1.0, 0.2, 0.2, 0.5],
        **args,
      )
      count += 1
  return spec


# -- Dual-racket G1 -----------------------------------------------------------
LEFT_RACKET_LINK = "left_racket_link"
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_RACKET_SITE = "right_racket_surface"
LEFT_RACKET_SITE = "left_racket_surface"
# HOPE DualHandRacketTargetCommandCfg.left_mount_quat: Rx(pi) about the handle,
# so the left red-face normal (site +Y) is left-wrist -Y, mirroring the right.
LEFT_MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)


def urdf_has_left_racket(urdf):
  root = ET.parse(urdf).getroot()
  return any(link.get("name") == LEFT_RACKET_LINK for link in root.findall("link"))


def urdf_fixed_subtree_inertial(urdf, link_name):
  """Mass, COM and COM inertia of a link plus its fixed-joint descendants.

  Expressed in ``link_name``'s frame, which is how the MuJoCo conversion
  merges fixed URDF links into their articulated body.
  """
  root = ET.parse(urdf).getroot()
  links = {link.get("name"): link for link in root.findall("link")}
  children = {}
  for joint in root.findall("joint"):
    children.setdefault(joint.find("parent").get("link"), []).append(joint)
  parts = []

  def visit(name, position, quaternion):
    inertial = links[name].find("inertial")
    if inertial is not None:
      p, q = urdf_pose(inertial.find("origin"))
      r = rotation(quaternion_multiply(quaternion, q))
      e = inertial.find("inertia")
      local = np.array([
        [float(e.get("ixx")), float(e.get("ixy")), float(e.get("ixz"))],
        [float(e.get("ixy")), float(e.get("iyy")), float(e.get("iyz"))],
        [float(e.get("ixz")), float(e.get("iyz")), float(e.get("izz"))],
      ])
      parts.append((
        float(inertial.find("mass").get("value")),
        position + rotation(quaternion) @ p,
        r @ local @ r.T,
      ))
    for joint in children.get(name, []):
      if joint.get("type") != "fixed":
        continue
      p, q = urdf_pose(joint.find("origin"))
      visit(
        joint.find("child").get("link"),
        position + rotation(quaternion) @ p,
        quaternion_multiply(quaternion, q),
      )

  visit(link_name, np.zeros(3), np.array([1.0, 0, 0, 0]))
  mass = sum(m for m, _, _ in parts)
  com = sum(m * c for m, c, _ in parts) / mass
  inertia = np.zeros((3, 3))
  for m, c, i in parts:
    d = c - com
    inertia += i + m * (d @ d * np.eye(3) - np.outer(d, d))
  return mass, com, inertia


def matrix_quaternion(r):
  """WXYZ quaternion of a proper rotation matrix."""
  w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
  x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
  y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
  z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
  x = np.copysign(x, r[2, 1] - r[1, 2])
  y = np.copysign(y, r[0, 2] - r[2, 0])
  z = np.copysign(z, r[1, 0] - r[0, 1])
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def set_body_inertial(body, mass, com, inertia):
  values, vectors = np.linalg.eigh(inertia)
  if np.linalg.det(vectors) < 0:
    vectors[:, 0] = -vectors[:, 0]
  body.mass = mass
  body.ipos = com
  body.iquat = matrix_quaternion(vectors)
  body.inertia = values
  body.fullinertia = [np.nan, 0, 0, 0, 0, 0]
  body.explicitinertial = True


def dual_racket_spec(base_factory, urdf):
  """Add the dual-racket URDF's left racket to the right-racket G1 asset.

  The left wrist's merged inertial is replaced by the URDF's (mount + racket
  instead of the fist), and a ``left_racket_surface`` site mirrors the asset's
  right racket site exactly as HOPE mirrors the right mount for the left hand.
  Visual meshes are not added; collisions come from training_collision_spec.
  """
  spec = base_factory()
  right = spec.site(RIGHT_RACKET_SITE)
  if not np.allclose(right.quat, [1, 0, 0, 0]):
    raise ValueError("Dual racket mirroring expects an identity right racket site")
  left_body = spec.body(LEFT_WRIST_BODY)
  set_body_inertial(left_body, *urdf_fixed_subtree_inertial(urdf, LEFT_WRIST_BODY))
  # Viewer only (group 2, no contact/mass): replace the fist by the right
  # mount/racket meshes under the same mirror (y -> -y, Rx(pi)). The racket
  # meshes are Z-symmetric, so this is their exact mirror image.
  for geom in list(left_body.geoms):
    if geom.name == "left_rubber_hand_0_visual":
      spec.delete(geom)
  for geom in list(spec.body("right_wrist_yaw_link").geoms):
    if not geom.name.startswith("right_racket") or not geom.name.endswith("_visual"):
      continue
    if not np.allclose(geom.quat, [1, 0, 0, 0]):
      raise ValueError(f"Dual racket mirroring expects identity visual {geom.name}")
    gx, gy, gz = geom.pos
    left_body.add_geom(
      name="left" + geom.name[len("right"):],
      type=geom.type,
      meshname=geom.meshname,
      pos=[gx, -gy, gz],
      quat=list(LEFT_MOUNT_QUAT),
      group=geom.group,
      contype=0,
      conaffinity=0,
      density=0.0,
      rgba=geom.rgba,
    )
  x, y, z = right.pos
  spec.body(LEFT_WRIST_BODY).add_site(
    name=LEFT_RACKET_SITE,
    pos=[x, -y, z],
    quat=list(LEFT_MOUNT_QUAT),
    size=right.size,
    group=right.group,
    rgba=right.rgba,
  )
  return spec

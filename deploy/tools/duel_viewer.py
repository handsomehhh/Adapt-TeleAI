"""One read-only scene for the canonical ball, table and two measured G1s.

Inputs are local snapshots, not robot DDS or command sockets. MuJoCo is used
only for forward kinematics of recorded measured joint angles, never stepping.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from deploy.io.viewer_snapshot import boot_id
from deploy.tools.real_viewer import DisplayState, finite_vector


def canonical_position(point, side, length=2.74, width=1.525):
  p = np.asarray(point, dtype=float)
  if side == "P1":
    return p.copy()
  if side != "P2":
    raise ValueError("Unknown robot side")
  return np.array([length - p[0], -width - p[1], p[2]])


def read_snapshot(path):
  try:
    with Path(path).open() as stream:
      value = json.loads(stream.read(65537))
    return value if isinstance(value, dict) else None
  except (OSError, ValueError):
    return None


def robot_state(snapshot, side, now, boot):
  state = DisplayState()
  if not snapshot or snapshot.get("schema") != "dual_robot_view_v1":
    return state
  if snapshot.get("robot_id") != side or snapshot.get("boot_id") != boot:
    return state
  t = snapshot.get("time")
  row = snapshot.get("data")
  if not isinstance(t, (int, float)) or not math.isfinite(t) or not isinstance(row, dict):
    return state
  # Validate age types before passing them to the original viewer validity gate.
  if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])
         for k in ("root_age", "dds_state_age")):
    return state
  state.apply(dict(kind="state", time=t, data=row))
  if not state.robot_valid(now):
    return DisplayState()
  return state


def ball_position(snapshot, now, boot):
  if not snapshot or snapshot.get("schema") != "dual_ball_view_v1":
    return None
  if snapshot.get("boot_id") != boot or snapshot.get("frame") != "canonical_P1":
    return None
  t = snapshot.get("time")
  if (not isinstance(t, (int, float)) or not math.isfinite(t)
      or not 0 <= now - t <= 0.2 or not finite_vector(snapshot.get("position"), 3)):
    return None
  return np.asarray(snapshot["position"])


def prediction_points(snapshot, now, boot):
  if (not snapshot or snapshot.get('schema') != 'dual_prediction_view_v1'
      or snapshot.get('boot_id') != boot or snapshot.get('frame') != 'canonical_P1'
      or snapshot.get('valid') is not True or snapshot.get('receiving_side') not in ('P1', 'P2')):
    return None
  t = snapshot.get('time')
  if not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= now - t <= .2:
    return None
  try:
    points = np.asarray(snapshot['points'], dtype=float)
    if points.ndim == 2 and points.shape[1] == 3 and 2 <= len(points) <= 180 and np.isfinite(points).all():
      return points
  except (KeyError, TypeError, ValueError):
    pass
  return None


class RacketTargetGhost:
  """Translate the actual racket visuals to the controller's selected goal.

  Only position is commanded by this overlay: orientation stays measured.
  Collision proxies are deliberately excluded to avoid overlapping faces.
  """

  def __init__(self, server, model, side, length, width):
    import mujoco
    from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh
    from deploy.tools.command_ghosts import command_geometry

    self.side, self.length, self.width = side, length, width
    self.handles = {}
    self.label = server.scene.add_label(f'/duel/{side}/target/label', f'{side} 球拍目标', visible=False)
    definition = command_geometry(model)
    if definition is None:
      return
    self.site = model.site(definition['racket_site']).id
    color = (80, 160, 245) if side == 'P1' else (245, 150, 60)
    for name in definition['racket_geoms']:
      geom = model.geom(name).id
      if model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0 or model.geom_rgba[geom, 3] <= 0:
        continue
      mesh = (mujoco_mesh_to_trimesh(model, geom)
              if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
              else create_primitive_mesh(model, geom))
      self.handles[geom] = server.scene.add_mesh_simple(
        f'/duel/{side}/target/geom_{geom}', mesh.vertices, mesh.faces,
        color=color, opacity=.35, cast_shadow=False, receive_shadow=False, visible=False)

  def update(self, data, row, offset, *, visible):
    from viser.transforms import SO3

    visible = bool(visible and self.handles and row.get('policy') == 'pingpong'
                   and finite_vector(row.get('goal_position'), 3))
    self.label.visible = visible
    for handle in self.handles.values():
      handle.visible = visible
    if not visible:
      return
    goal = np.asarray(row['goal_position'])
    delta = goal - offset - data.site_xpos[self.site]
    rotate = np.diag([-1., -1., 1.]) if self.side == 'P2' else np.eye(3)
    for geom, handle in self.handles.items():
      handle.position = canonical_position(data.geom_xpos[geom] + delta + offset,
                                           self.side, self.length, self.width)
      handle.wxyz = SO3.from_matrix(rotate @ data.geom_xmat[geom].reshape(3, 3)).wxyz
    self.label.position = canonical_position(goal, self.side, self.length, self.width) + [0, 0, .14]

  def remove(self):
    for handle in self.handles.values():
      handle.remove()
    self.label.remove()


class RecordedRobot:
  def __init__(self, server, directory, side, boot, length, width):
    import mujoco
    from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

    self.directory, self.side = Path(directory), side
    self.length, self.width = length, width
    meta = json.loads((self.directory / "metadata.json").read_text())
    if meta.get("robot_id") != side or meta.get("boot_id") != boot:
      raise ValueError("需要本次开机、对应侧的新双机可视化记录")
    model_path = self.directory / "scene.mjb"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta["model_sha256"]:
      raise ValueError("模型文件校验未通过")
    self.model = mujoco.MjModel.from_binary_path(str(model_path))
    self.data = mujoco.MjData(self.model)
    free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or len(meta["joint_names"]) != 29:
      raise ValueError("Expected one free root and 29 measured robot joints")
    self.root = self.model.jnt_qposadr[free[0]]
    self.addresses = [self.model.jnt_qposadr[self.model.joint("robot/" + n).id]
                      for n in meta["joint_names"]]
    self.offset = np.asarray(meta["world_translation"], dtype=float)
    if not finite_vector(self.offset, 3):
      raise ValueError("Invalid recorded model/world offset")
    # Select the free-root subtree only: never draw another table, floor or ball.
    root_body = self.model.jnt_bodyid[free[0]]
    bodies = {int(root_body)}
    for body in range(root_body + 1, self.model.nbody):
      if int(self.model.body_parentid[body]) in bodies:
        bodies.add(body)
    self.handles = {}
    color = (80, 160, 245) if side == "P1" else (245, 150, 60)
    for geom in range(self.model.ngeom):
      # Collision proxies overlap the visible racket faces; drawing both causes
      # gray occlusion / z-fighting. Keep all collision geometry out of the view.
      visual = self.model.geom_contype[geom] == 0 and self.model.geom_conaffinity[geom] == 0
      if self.model.geom_bodyid[geom] not in bodies or not visual:
        continue
      if self.model.geom_rgba[geom, 3] <= 0:
        continue
      mesh = (mujoco_mesh_to_trimesh(self.model, geom)
              if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
              else create_primitive_mesh(self.model, geom))
      mesh_id = self.model.geom_dataid[geom]
      mesh_name = (self.model.mesh(mesh_id).name or ""
                   if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else "")
      mesh_name = mesh_name.rsplit("/", 1)[-1]
      # Head and logo share torso_link's body, so select the torso mesh itself.
      if mesh_name == "torso_link" or mesh_name.startswith("torso_link_rev_"):
        self.handles[geom] = server.scene.add_mesh_simple(
          f"/duel/{side}/geom_{geom}", mesh.vertices, mesh.faces,
          color=color, visible=False,
        )
      else:
        # The conversion preserves XML material/geom RGBA and textures.
        self.handles[geom] = server.scene.add_mesh_trimesh(
          f"/duel/{side}/geom_{geom}", mesh, visible=False,
        )
    self.label = server.scene.add_label(f"/duel/{side}/label", side, visible=False)
    self.ghost = RacketTargetGhost(server, self.model, side, length, width)

  def update(self, state, now, *, show_target=True):
    import mujoco
    from viser.transforms import SO3

    valid = state.robot_valid(now)
    for handle in self.handles.values():
      handle.visible = valid
    self.label.visible = valid
    if not valid:
      self.ghost.update(self.data, {}, self.offset, visible=False)
      return
    row = state.robot
    self.data.qpos[self.addresses] = row["q"]
    self.data.qpos[self.root:self.root + 3] = np.asarray(row["root_pos"]) - self.offset
    q = np.asarray(row["root_quat"])
    self.data.qpos[self.root + 3:self.root + 7] = q / np.linalg.norm(q)
    mujoco.mj_kinematics(self.model, self.data)
    rotate = np.diag([-1., -1., 1.]) if self.side == "P2" else np.eye(3)
    for geom, handle in self.handles.items():
      handle.position = canonical_position(
        self.data.geom_xpos[geom] + self.offset, self.side, self.length, self.width)
      handle.wxyz = SO3.from_matrix(rotate @ self.data.geom_xmat[geom].reshape(3, 3)).wxyz
    self.label.position = canonical_position(
      row["root_pos"], self.side, self.length, self.width) + [0, 0, 0.65]
    self.ghost.update(self.data, row, self.offset, visible=show_target)

  def remove(self):
    for handle in self.handles.values():
      handle.remove()
    self.label.remove()
    self.ghost.remove()


class DuelScene:
  def __init__(self, runtime_dir, ball_snapshot, *, host="127.0.0.1", port=8082,
               length=2.74, width=1.525, planner_snapshot=None):
    import viser

    self.server = viser.ViserServer(host=host, port=port, label="G1 双机实时观测")
    if self.server.get_port() != port:
      self.server.stop()
      raise RuntimeError(f"Port {port} is occupied; choose another --web-port")
    self.runtime_dir, self.ball_snapshot = Path(runtime_dir), Path(ball_snapshot)
    self.planner_snapshot = Path(planner_snapshot) if planner_snapshot is not None else None
    self.boot, self.length, self.width = boot_id(), length, width
    self.robots, self.errors = {}, {}
    self.sources = {}
    self.retry_at = {"P1": 0., "P2": 0.}
    self.server.scene.set_up_direction("+z")
    self.server.scene.add_grid(
      "/ground", infinite_grid=True, fade_distance=50.0,
      shadow_opacity=0.2, plane_opacity=0.4, position=(0, 0, -.76),
    )
    self.server.scene.add_box("/table/top", dimensions=(length, width, .05),
                             position=(length / 2, -width / 2, -.025), color=(28, 86, 125))
    self.server.scene.add_box("/table/net", dimensions=(.008, width, .1525),
                             position=(length / 2, -width / 2, .07625), color=(210, 210, 210),
                             opacity=.6)
    for i, (x, y) in enumerate(((.2, -.2), (.2, -width + .2),
                                (length - .2, -.2), (length - .2, -width + .2))):
      self.server.scene.add_box(f"/table/leg_{i}", dimensions=(.05, .05, .71),
                               position=(x, y, -.405), color=(100, 100, 100))
    self.ball = self.server.scene.add_icosphere("/ball", radius=.02,
                                               color=(255, 255, 255), visible=False)
    self.prediction = self.server.scene.add_line_segments(
      '/planner/prediction', points=np.zeros((1, 2, 3), dtype=np.float32),
      colors=(255, 225, 70), line_width=3., visible=False)
    self.show_prediction = self.server.gui.add_checkbox('显示 planner 预测轨迹', initial_value=True)
    self.show_targets = self.server.gui.add_checkbox('显示两侧球拍目标虚影', initial_value=True)
    self.status = self.server.gui.add_markdown("等待 P1/P2 实时数据…")
    self.server.gui.add_markdown(
      "Torso：P1 蓝色 · P2 橙色，其余部位保留模型原色。"
      "球台顶面 Z=0，地面 Z=-0.76，统一使用 P1 桌面坐标。\n\n"
      "机器人显示实测关节和控制器采用的 pelvis 位姿；超过 0.2 秒的数据隐藏。\n\n"
      "黄色线：planner 本次求解的来球预测轨迹。蓝/橙色半透明球拍：P1/P2 控制器实际采用的目标位置；"
      "虚影姿态沿用实测姿态，仅表示目标位置。无有效目标或观测过期时隐藏。\n\n"
      "此页面只读，关闭它不会停止机器人。")

    @self.server.on_client_connect
    def camera(client):
      client.camera.up_direction = (0, 0, 1)
      client.camera.position = (length / 2 + 3.8, 3.1, 2.3)
      client.camera.look_at = (length / 2, -width / 2, .15)

  def update(self, now):
    status = []
    for side in ("P1", "P2"):
      # Resolve once per update so model and state always come from the same
      # actor. Each role can restart independently while this viewer stays open.
      directory = (self.runtime_dir / side / "policy/visualization").resolve()
      if self.sources.get(side) != directory:
        if side in self.robots:
          self.robots.pop(side).remove()
        self.sources[side] = directory
        self.errors.pop(side, None)
        self.retry_at[side] = 0.
      if side not in self.robots and now >= self.retry_at[side]:
        self.retry_at[side] = now + 1.
        try:
          self.robots[side] = RecordedRobot(self.server, directory, side, self.boot,
                                             self.length, self.width)
          self.errors.pop(side, None)
        except (OSError, ValueError, KeyError) as exc:
          self.errors[side] = str(exc)
      state = robot_state(read_snapshot(directory / "latest_state.json"), side, now, self.boot)
      if side in self.robots:
        self.robots[side].update(state, now, show_target=self.show_targets.value)
      valid = side in self.robots and state.robot_valid(now)
      status.append(f"**{side}**：" + ("实时" if valid else "等待 / 数据过期，已隐藏"))
      if self.errors.get(side):
        status.append(f"{side}：{self.errors[side]}")
      if side in self.robots:
        status.append(f"**{side} 球拍目标**：" + ('显示中' if self.robots[side].ghost.label.visible else '无有效目标 / 已隐藏'))
    ball = ball_position(read_snapshot(self.ball_snapshot), now, self.boot)
    self.ball.visible = ball is not None
    if ball is not None:
      self.ball.position = ball
    status.append("**球**：" + ("实时动捕" if ball is not None else "等待 / 数据过期，已隐藏"))
    snapshot = read_snapshot(self.planner_snapshot) if self.planner_snapshot is not None else None
    points = prediction_points(snapshot, now, self.boot)
    self.prediction.visible = bool(self.show_prediction.value and points is not None)
    if self.prediction.visible:
      self.prediction.points = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
    status.append('**planner 预测**：' + (f"{snapshot['receiving_side']} 来球" if self.prediction.visible
                                       else '无有效预测 / 已隐藏'))
    self.status.content = "\n\n".join(status)

  def close(self):
    self.server.stop()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runtime-dir", type=Path, required=True)
  parser.add_argument("--ball-snapshot", type=Path, required=True)
  parser.add_argument("--planner-snapshot", type=Path)
  parser.add_argument("--web-host", default="127.0.0.1")
  parser.add_argument("--web-port", type=int, default=8082)
  parser.add_argument("--table-length", type=float, default=2.74)
  parser.add_argument("--table-width", type=float, default=1.525)
  args = parser.parse_args()
  if not 1 <= args.web_port <= 65535:
    parser.error("Valid web port required")
  scene = DuelScene(args.runtime_dir, args.ball_snapshot, host=args.web_host, port=args.web_port,
                    length=args.table_length, width=args.table_width, planner_snapshot=args.planner_snapshot)
  try:
    while True:
      scene.update(time.monotonic())
      time.sleep(1 / 30)
  except KeyboardInterrupt:
    pass
  finally:
    scene.close()


if __name__ == "__main__":
  main()

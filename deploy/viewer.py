"""Presentation-only MuJoCo viewers shared by deployment simulations and replays.

No policy, physics, sampling or evaluation logic belongs here. Browser controls
are queued and dispatched by the simulation/replay thread, just like native keys.
"""

import copy
import math
import queue
import threading
import time
from dataclasses import dataclass
from itertools import zip_longest

import mujoco
import numpy as np


@dataclass
class ViewerSettings:
  backend: str = "web"
  host: str = "127.0.0.1"
  port: int = 8080
  wait: bool = True


settings = ViewerSettings()


def add_viewer_arguments(parser):
  parser.add_argument(
    "--viewer",
    choices=("web", "native"),
    default="web",
    help="Visualization backend; simulation still needs --render",
  )
  parser.add_argument("--web-host", default="127.0.0.1")
  parser.add_argument("--web-port", type=int, default=8080)
  parser.add_argument(
    "--web-no-wait",
    action="store_true",
    help="Start without waiting for the first browser connection",
  )


def configure_viewer(args):
  if not 1 <= args.web_port <= 65535:
    raise ValueError("web-port must be within 1..65535")
  global settings
  settings = ViewerSettings(
    args.viewer, args.web_host, args.web_port, not args.web_no_wait
  )


def overlay_batches(scene):
  """Convert existing mjvGeom overlays without recomputing their semantics.

  mjvGeom sizes are normalized XYZ extents, unlike mjModel primitive sizes.
  Connector arrows originate at pos; capsules are centered there.
  """
  batches = {}
  for i in range(scene.ngeom):
    geom = scene.geoms[i]
    kind = int(geom.type)
    if float(geom.rgba[3]) <= 0:
      continue
    names = {
      int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
      int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "sphere",
      int(mujoco.mjtGeom.mjGEOM_BOX): "box",
      int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
      int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
      int(mujoco.mjtGeom.mjGEOM_ARROW): "arrow",
    }
    name = names.get(kind)
    if name is None:
      raise ValueError(f"Unsupported debug overlay geometry {kind}")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, geom.mat.ravel())
    pos, scale, rgba = geom.pos.copy(), geom.size.copy(), geom.rgba.copy()

    def append(shape, position, size, quat=quat, rgba=rgba):
      batches.setdefault(shape, []).append((position, quat, size, rgba))

    if name == "capsule":
      append("cylinder", pos, scale)
      offset = geom.mat[:, 2] * scale[2]
      for point in (pos - offset, pos + offset):
        append("sphere", point, np.full(3, scale[0]))
    else:
      append(name, pos, scale)
  return {
    name: tuple(np.array([item[j] for item in values]) for j in range(4))
    for name, values in batches.items()
  }


class WebMujocoViewer:
  """Subset of the passive viewer interface consumed by existing test runners."""

  def __init__(
    self, model, data, key_callback=None, *, host="127.0.0.1", port=8080, wait=True, **_
  ):
    import trimesh
    import viser
    from mjviser import ViserMujocoScene

    self.data = data
    self._mutex = threading.RLock()
    self._actions = queue.SimpleQueue()
    self._running = True
    self._callback = key_callback
    self._owner = getattr(key_callback, "__self__", None)
    self._timeline = None
    self._last_sync = -math.inf
    self._texts = None
    self._last_texts = None
    self._camera_ready = False
    self._handles = {}
    self.user_scn = mujoco.MjvScene(model, maxgeom=10000)
    self.cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(self.cam)
    self.cam.lookat[:] = model.stat.center
    self.cam.distance = 4.5
    self.cam.azimuth, self.cam.elevation = 125, -20
    self.opt = mujoco.MjvOption()
    self.server = viser.ViserServer(host=host, port=port, label="乒乓球测试平台")
    try:
      self.server.gui.configure_theme(control_width="large", show_share_button=False)
      if hasattr(self.server.gui, "main_panel"):
        self.server.gui.main_panel.dock_right()
        self.server.gui.main_panel.set_width(520)
      # mjviser changes visual defaults on construction. Keep them off the plant.
      self.scene = ViserMujocoScene(self.server, copy.copy(model), num_envs=1)
      self.scene.camera_tracking_enabled = False
      self._meshes = {
        "sphere": trimesh.creation.icosphere(subdivisions=2),
        "box": trimesh.creation.box(extents=[2, 2, 2]),
        "cylinder": trimesh.creation.cylinder(radius=1, height=2, sections=16),
      }
      shaft = trimesh.creation.cylinder(radius=1, height=0.8, sections=12)
      shaft.apply_translation([0, 0, 0.4])
      head = trimesh.creation.cone(radius=2.5, height=0.2, sections=12)
      head.apply_translation([0, 0, 0.8])
      self._meshes["arrow"] = trimesh.util.concatenate([shaft, head])
      tabs = self.server.gui.add_tab_group()
      with tabs.add_tab("控制"):
        self.server.gui.add_markdown("播放与定位 · 支持直接输入和键盘快捷键")
        self._build_controls()
        self._build_hotkeys()
        with self.server.gui.add_folder("视角与结束", expand_by_default=False):
          self._button("重置视角", lambda: self._set_cameras())
          self._button("结束查看 / 停止测试", self._stop)
      with tabs.add_tab("状态与误差"):
        self._status = self.server.gui.add_markdown("等待场景状态…")
      with tabs.add_tab("场景显示"):
        self._fov = self.server.gui.add_slider(
          "视野角（°）", min=20, max=100, step=1, initial_value=45
        )

        @self._fov.on_update
        def change_fov(event):
          if event.client is not None:
            value = float(self._fov.value)
            self._actions.put(lambda: self._set_fov(value))

        with self.server.gui.add_folder("几何分组", expand_by_default=False):
          self.scene.create_groups_gui()
        with self.server.gui.add_folder("坐标与接触显示", expand_by_default=False):
          self.scene.create_overlay_gui()

      @self.server.on_client_connect
      def connected(client):
        self._set_camera(client)
        # Viser has no public script API. Use its own script message only for
        # forwarding parent-page keys to the standard Viser command hotkeys.
        import json
        from pathlib import Path

        from viser import _messages

        stylesheet = Path(__file__).with_name("viewer.css").read_text()
        source = (
          "const style = document.createElement('style'); style.textContent = "
          + json.dumps(stylesheet)
          + "; document.head.appendChild(style); document.body.classList.add('pingpong-viewer');\n"
          + Path(__file__)
          .with_name("viewer_keyboard.js")
          .read_text()
          .replace("__REPLAY_MODE__", "true" if self._timeline is not None else "false")
        )
        client._websock_connection.queue_message(
          _messages.RunJavascriptMessage(source=source)
        )

      self.scene.update_from_mjdata(data)
      print(
        f"[Web viewer] http://{host}:{self.server.get_port()}  "
        "(remote: forward this port over SSH)",
        flush=True,
      )
      if wait:
        print(
          "[Web viewer] Waiting for browser; no simulation steps yet. Ctrl-C cancels.",
          flush=True,
        )
        while self._running and not self.server.get_clients():
          self._drain_actions()
          time.sleep(0.05)
    except BaseException:
      self.server.stop()
      raise

  def _button(self, label, action):
    handle = self.server.gui.add_button(label)
    handle.on_click(lambda _: self._actions.put(action))
    return handle

  def _key(self, code):
    if self._callback is not None:
      self._callback(code)

  def _key_group(self, label, pairs):
    codes = dict(pairs)
    handle = self.server.gui.add_button_group(label, options=list(codes))

    @handle.on_click
    def clicked(event):
      code = codes[event.target.value]
      self._actions.put(lambda: self._key(code))

  def _build_controls(self):
    owner = self._owner
    is_replay = (
      owner is not None and hasattr(owner, "recording") and hasattr(owner, "position")
    )
    self._timeline = owner if is_replay else None
    self._speed = (
      getattr(owner, "speed", None)
      or getattr(owner, "playback", None)
      or (owner if hasattr(owner, "set_rate") else None)
    )
    with self.server.gui.add_folder("播放", expand_by_default=True):
      if owner is not None and hasattr(owner, "paused"):
        self._play_state = self.server.gui.add_markdown("**状态：已暂停**")
        self._play_button = self._button("继续播放 · Space", lambda: self._key(32))
      if self._speed is not None:
        self._rate_display = self.server.gui.add_markdown(
          f"**当前倍速：{self._speed.rate:g}×**"
        )
        speed = self._rate_input = self.server.gui.add_number(
          "播放倍速（可直接输入）",
          initial_value=float(self._speed.rate),
          min=0.0625,
          max=max(8.0, float(self._speed.rate)),
          step=0.0625,
        )
        self._published_rate = float(self._speed.rate)

        @speed.on_update
        def change_speed(event):
          if event.client is not None:
            value = float(speed.value)
            if math.isfinite(value) and 0.0625 <= value <= max(
              8.0, self._published_rate
            ):
              self._actions.put(lambda: self._current_speed().set_rate(value))

        speed_buttons = self.server.gui.add_button_group(
          "快捷调速", options=["减半 ÷2", "正常 1×", "加倍 ×2"]
        )

        @speed_buttons.on_click
        def adjust_speed(event):
          code = {"减半 ÷2": 91, "正常 1×": 92, "加倍 ×2": 93}[event.target.value]
          self._actions.put(lambda: self._key(code))

    if is_replay:
      with self.server.gui.add_folder("TTS 时序", expand_by_default=True):
        self._tts_display = self.server.gui.add_markdown("等待执行 command…")
      with self.server.gui.add_folder("按帧定位", expand_by_default=True):
        self._frame_count = self.server.gui.add_number(
          "每次跳转帧数（可直接输入）",
          initial_value=1,
          min=1,
          max=max(1, len(owner.recording.times)),
          step=1,
        )
        count_buttons = self.server.gui.add_button_group(
          "调整帧数", options=["减少 −1", "增加 +1"]
        )

        @count_buttons.on_click
        def adjust_count(event):
          delta = -1 if event.target.value == "减少 −1" else 1
          self._frame_count.value = max(
            1, min(len(owner.recording.times), int(self._frame_count.value) + delta)
          )

        frame_buttons = self.server.gui.add_button_group(
          "执行跳转", options=["后退", "前进"]
        )

        @frame_buttons.on_click
        def frame_step(event):
          self._queue_frame_step(-1 if event.target.value == "后退" else 1)

      with self.server.gui.add_folder("时间与阶段", expand_by_default=True):
        self._key_group("按秒跳转", [("后退 1 秒", 263), ("前进 1 秒", 262)])
        self._key_group(
          "按阶段跳转", [("上一阶段 · P", ord("P")), ("下一阶段 · N", ord("N"))]
        )
        duration = owner.recording.duration
        self._seek = self.server.gui.add_slider(
          "回放时间（秒）",
          min=0,
          max=max(duration, 0.001),
          step=0.001,
          initial_value=owner.position,
        )
        self._published_seek = owner.position

        @self._seek.on_update
        def seek(event):
          if event.client is not None:
            value = float(self._seek.value)

            def apply():
              with owner.lock:
                owner.position = value
                owner.paused = True

            self._actions.put(apply)

      with self.server.gui.add_folder("更多操作", expand_by_default=False):
        self._button("从头播放 · R", lambda: self._key(ord("R")))
    elif owner is not None and hasattr(owner, "step_requested"):
      for label, code in [
        ("START / 定位", 257),
        ("悬挂下放（L）", 76),
        ("AMP（A）", 65),
        ("乒乓策略 UP（W）", 87),
        ("返回 AMP（S）", 83),
        ("发一球（F）", 70),
        ("连续发球（T）", 84),
        ("单步（N）", 78),
        ("重置（R）", 82),
        ("停止动作（B）", 66),
      ]:
        self._button(label, lambda code=code: self._key(code))

  def _queue_frame_step(self, direction):
    count = max(1, int(self._frame_count.value))
    self._actions.put(lambda: self._timeline.step_frames(direction * count))

  def _current_speed(self):
    # A simulation reset replaces PlaybackControl while keeping this viewer.
    return (
      getattr(self._owner, "speed", None)
      or getattr(self._owner, "playback", None)
      or self._speed
    )

  def _build_hotkeys(self):
    keys = []
    if self._speed is not None:
      keys += [(key, ord(key), "播放速度") for key in "[]-=\\"]
      keys += [("=", ord("+"), "加速", "shift")]
    if self._owner is not None and hasattr(self._owner, "paused"):
      keys += [("space", 32, "播放 / 暂停")]
    if self._timeline is not None:
      keys += [
        ("R", 82, "重新播放"),
        (",", 44, "上一帧"),
        (".", 46, "下一帧"),
        ("N", 78, "下一阶段"),
        ("P", 80, "上一阶段"),
        ("arrowleft", 263, "后退一秒"),
        ("arrowright", 262, "前进一秒"),
        ("home", 268, "回到开头"),
        ("end", 269, "跳到结尾"),
      ]
    elif self._owner is not None and hasattr(self._owner, "step_requested"):
      keys += [(key, ord(key), "仿真控制") for key in "LAW SFTNRBXQPH".replace(" ", "")]
      keys += [("enter", 257, "START")]
    self._hotkeys = []
    for entry in keys:
      key, code, label, *modifier = entry
      command = self.server.gui.add_command(
        label, hotkey=key, modifier=modifier[0] if modifier else None
      )
      command.on_trigger(
        lambda _, code=code: self._actions.put(lambda: self._key(code))
      )
      self._hotkeys.append(command)

  def _set_camera(self, client):
    az, el = np.deg2rad([self.cam.azimuth, self.cam.elevation])
    client.camera.up_direction = (0, 0, 1)
    client.camera.fov = np.deg2rad(float(self._fov.value))
    client.camera.position = self.cam.lookat + self.cam.distance * np.array(
      [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)]
    )
    client.camera.look_at = self.cam.lookat

  def _set_fov(self, degrees):
    for client in self.server.get_clients().values():
      client.camera.fov = np.deg2rad(degrees)

  def _set_cameras(self):
    for client in self.server.get_clients().values():
      self._set_camera(client)

  def _stop(self):
    self._running = False
    owner = self._owner
    if owner is not None and hasattr(owner, "paused"):
      owner.paused = False
    self._key(ord("Q"))

  def _drain_actions(self):
    while not self._actions.empty():
      self._actions.get_nowait()()

  def lock(self):
    return self._mutex

  def set_texts(self, texts):
    self._texts = texts

  def is_running(self):
    self._drain_actions()
    return self._running

  def sync(self):
    self._drain_actions()
    now = time.monotonic()
    if now - self._last_sync < 1 / 30:
      return
    self._last_sync = now
    with self._mutex, self.server.atomic():
      if self._owner is not None and hasattr(self._owner, "paused"):
        paused = self._owner.paused
        self._play_state.content = (
          "**状态：已暂停 ⏸**" if paused else "**状态：播放中 ▶**"
        )
        self._play_button.label = "继续播放 · Space" if paused else "暂停播放 · Space"
      if self._timeline is not None:
        status = self._timeline.tts_status()
        if status is None:
          self._tts_display.content = "等待有效 command / 缺少 TTS 时间记录"
        else:
          remaining = status["remaining"]
          end = (
            f"倒计时 **{remaining:.3f} s**"
            if remaining >= 0
            else f"已超时 **{-remaining:.3f} s**"
          )
          text = (
            f"开始（接收）：**{status['start']:.3f} s** · 已过 {status['elapsed']:.3f} s\n\n"
            f"初始 TTS：**{status['initial']:.3f} s**\n\n"
            f"结束（deadline）：**{status['deadline']:.3f} s** · {end}"
          )
          if "event_error" in status:
            label = (
              "估计击球" if status["event_kind"] == "estimated_hit" else "实测穿平面"
            )
            delta = status["event_error"] * 1000
            text += (
              f"\n\n{label}：{status['event_time']:.3f} s · "
              f"相对 deadline **{'滞后' if delta >= 0 else '提前'} {abs(delta):.1f} ms**"
            )
          self._tts_display.content = text
      if self._speed is not None:
        rate = self._current_speed().rate
        self._rate_display.content = f"**当前倍速：{rate:g}×**"
        # Preserve browser edits until their queued callback has been applied.
        if self._rate_input.value == self._published_rate:
          self._published_rate = rate
          self._rate_input.value = rate
        elif self._rate_input.value == rate:
          self._published_rate = rate
      if not self._camera_ready:
        self._set_cameras()
        self._camera_ready = True
      self.scene.update_from_mjdata(self.data)
      batches = overlay_batches(self.user_scn)
      for name, handle in self._handles.items():
        if name not in batches:
          handle.visible = False
      for name, (positions, rotations, scales, colors) in batches.items():
        handle = self._handles.get(name)
        if handle is not None and len(handle.batched_positions) != len(positions):
          handle.remove()
          handle = None
        if handle is None:
          mesh = self._meshes[name]
          handle = self.server.scene.add_batched_meshes_simple(
            "/evaluation/" + name,
            mesh.vertices,
            mesh.faces,
            batched_positions=positions,
            batched_wxyzs=rotations,
            batched_scales=scales,
            batched_colors=(colors[:, :3] * 255).astype(np.uint8),
            batched_opacities=colors[:, 3],
            cast_shadow=False,
            receive_shadow=False,
          )
          self._handles[name] = handle
        else:
          handle.visible = True
          handle.batched_positions, handle.batched_wxyzs = positions, rotations
          handle.batched_scales = scales
          handle.batched_colors = (colors[:, :3] * 255).astype(np.uint8)
          handle.batched_opacities = colors[:, 3]
      if self._texts != self._last_texts:
        self._last_texts = self._texts
        text = self._texts or (None, None, "", "")
        lines = ["| 指标 | 当前值 |", "|---|---|"]
        for label, value in zip_longest(
          str(text[2] or "").splitlines(), str(text[3] or "").splitlines(), fillvalue=""
        ):
          lines.append(f"| {label.replace('|', '/')} | {value.replace('|', '/')} |")
        self._status.content = "\n".join(lines)
      if self._timeline is not None:
        # A browser update reaches the GUI value before its worker callback is
        # dispatched. Do not overwrite that pending seek with the old clock.
        if self._seek.value == self._published_seek:
          self._published_seek = self._timeline.position
          self._seek.value = self._published_seek
        elif self._seek.value == self._timeline.position:
          self._published_seek = self._timeline.position

  def close(self):
    self._running = False
    self.server.stop()

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()


def launch_passive(model, data, **kwargs):
  if settings.backend == "native":
    from mujoco.viewer import launch_passive as native

    return native(model, data, **kwargs)
  return WebMujocoViewer(
    model, data, host=settings.host, port=settings.port, wait=settings.wait, **kwargs
  )

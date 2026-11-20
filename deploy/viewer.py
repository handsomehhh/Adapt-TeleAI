"""Viser controls and overlays; callbacks queue commands for the simulation thread."""

import copy
import queue
import time
from pathlib import Path

import numpy as np
import viser
from mjviser import ViserMujocoScene
from viser import _messages

from deploy.comparison import ReferenceRobot
from deploy.control import MODES
from deploy.playback import DisplayFrames


class Viewer:
  def __init__(self, session, host="127.0.0.1", port=8085):
    self.session = session
    self.commands = queue.SimpleQueue()
    self.running, self.single_step, self.rate = True, False, 1.0
    self.frames = DisplayFrames(session)
    self.playback_stats = {"fps": 0.0, "realtime_factor": 0.0, "dropped_s": 0.0}
    self.last_status_update = -float("inf")
    self.server = viser.ViserServer(host=host, port=port, label="G1 AdaPT 部署验证")
    self.scene = ViserMujocoScene(
      self.server, copy.copy(session.world.model), num_envs=1
    )
    self.scene.camera_tracking_enabled = False
    self.reference_robot = ReferenceRobot(self.server.scene, session.world)
    self.actual_label = self.server.scene.add_label("/actual_label", "Tracker 实际执行")
    self.server.scene.set_up_direction("+z")
    gui = self.server.gui
    gui.add_markdown(
      "# G1 AdaPT 模块测试\nCPU MuJoCo · 冻结训练模型\n原色机器人为实际执行，绿色机器人为完整参考动作。"
      "并排显示只平移参考图像，误差始终在原坐标中计算。"
    )
    mode = gui.add_dropdown(
      "测试模块",
      options=("replay",) if session.mode == "replay" else MODES,
      initial_value=session.mode,
    )
    mode.on_update(lambda event: self.commands.put(("mode", event.target.value)))
    self.motion_picker, self.motion_time = None, None
    self.motion_buttons = []
    if session.mode != "replay":
      with gui.add_folder("Tracker motion", expand_by_default=True):
        selected = next(
          name
          for name, path in session.motion_catalog.items()
          if path == session.motion
        )
        self.motion_picker = gui.add_dropdown(
          "选择 motion", options=tuple(session.motion_catalog), initial_value=selected
        )
        self.motion_picker.on_update(
          lambda e: (
            self.commands.put(("motion", e.target.value))
            if e.client_id is not None
            else None
          )
        )
        for label, direction in (("上一个 motion", -1), ("下一个 motion", 1)):
          button = gui.add_button(label)
          button.on_click(
            lambda _e, d=direction: self.commands.put(("adjacent_motion", d))
          )
          self.motion_buttons.append(button)
        self.motion_time = gui.add_slider(
          "motion 时间 (s)",
          min=0.0,
          max=max(0.001, session.controller.reference.duration),
          step=0.001,
          initial_value=0.0,
        )
        self.motion_time.on_update(
          lambda e: (
            self.commands.put(("motion_seek", float(e.target.value)))
            if e.client_id is not None
            else None
          )
        )
        self.loop = gui.add_checkbox(
          "循环 motion（每轮重置机器人）", initial_value=False
        )
        self.loop.on_update(lambda e: self.commands.put(("loop", e.target.value)))
        self.motion_buttons.append(self.loop)
        gui.add_markdown("切换或拖动时间会重置实际机器人并暂停；取消暂停继续执行。")
    paused = gui.add_checkbox("暂停", initial_value=True)
    paused.on_update(
      lambda event: (
        self.commands.put(("pause", event.target.value))
        if event.client_id is not None
        else None
      )
    )
    self.pause_handle = paused
    self.hotkeys = {}
    for key, label, action in (
      ("space", "播放 / 暂停", "toggle_pause"),
      ("R", "重置机器人与参考", "reset"),
    ):
      command = gui.add_command(label, hotkey=key)
      command.on_trigger(lambda _event, name=action: self.commands.put((name, None)))
      self.hotkeys[key] = command
    gui.add_markdown(
      "快捷键：**Space 暂停 / 继续** · **R 从头重置并暂停**。先点击三维画面。"
    )
    for label, command in (
      ("单步", "step"),
      ("重置机器人与参考", "reset"),
      ("发一球", "feed"),
      ("保存记录", "save"),
      ("停止", "stop"),
    ):
      button = gui.add_button(label)
      button.on_click(lambda _event, name=command: self.commands.put((name, None)))
    scenario = gui.add_dropdown(
      "来球流程",
      options=("replay",)
      if session.mode == "replay"
      else ("independent", "continuous"),
      initial_value=session.scenario,
    )
    scenario.on_update(lambda e: self.commands.put(("scenario", e.target.value)))
    rate = gui.add_slider(
      "播放倍速（1× 为真实时间）", min=0.1, max=2.0, step=0.05, initial_value=1.0
    )
    rate.on_update(lambda e: self.commands.put(("rate", e.target.value)))
    speed = gui.add_slider(
      "动作速度 alpha（改变策略输入）",
      min=0.5,
      max=2.0,
      step=0.05,
      initial_value=session.speed,
    )
    speed.on_update(lambda e: self.commands.put(("speed", e.target.value)))
    self.speed_handle = speed
    self.ghost = gui.add_checkbox("显示参考骨架", initial_value=True)
    self.show_reference = gui.add_checkbox("显示完整参考机器人", initial_value=True)
    self.layout = gui.add_dropdown(
      "对比布局", options=("并排", "重叠"), initial_value="并排"
    )
    self.opacity = gui.add_slider(
      "参考机器人透明度", min=0.1, max=1.0, step=0.05, initial_value=0.65
    )
    with gui.add_folder("MVAE latent（仅手动生成模式）", expand_by_default=False):
      for index in range(32):
        handle = gui.add_slider(
          f"z{index}", min=-3.0, max=3.0, step=0.1, initial_value=0.0
        )
        handle.on_update(
          lambda e, i=index: self.commands.put(("latent", (i, e.target.value)))
        )
    self.status = gui.add_markdown("已加载，等待开始。")
    self.message = gui.add_markdown("")
    self.chart = gui.add_uplot(
      (np.array([0.0]), np.array([0.0]), np.array([0.0])),
      (
        {"label": "控制步"},
        {"label": "拍心误差 cm", "stroke": "#38bdf8"},
        {"label": "关节 RMS rad", "stroke": "#34d399"},
      ),
      title="最近控制步的跟踪误差",
      scales={"x": {"time": False}},
      height=180,
    )
    self.chart_counter = 0
    self.ball = self.server.scene.add_icosphere(
      "/ball", radius=0.02, color=(255, 130, 20)
    )
    self.actual = self.server.scene.add_icosphere(
      "/actual_racket", radius=0.018, color=(40, 140, 250)
    )
    self.reference = self.server.scene.add_icosphere(
      "/reference_racket", radius=0.018, color=(30, 220, 100)
    )
    self.target = self.server.scene.add_icosphere(
      "/landing_target", radius=0.035, color=(220, 60, 180)
    )
    self.target.position = (2.7, 0.0, 0.78)
    self.path = self.server.scene.add_line_segments(
      "/prediction",
      points=np.zeros((29, 2, 3)),
      colors=(230, 180, 30),
      thickness=2,
      thickness_units="screen",
    )
    self.normal = self.server.scene.add_line_segments(
      "/paddle_normal",
      points=np.zeros((1, 2, 3)),
      colors=(30, 170, 250),
      thickness=3,
      thickness_units="screen",
    )
    self.scene.update_from_mjdata(session.world.data)
    self.skeleton = self.server.scene.add_line_segments(
      "/reference_skeleton",
      points=np.zeros((29, 2, 3)),
      colors=(40, 210, 100),
      thickness=2,
      thickness_units="screen",
    )
    self.timeline = None
    if session.mode == "replay":
      self.timeline = gui.add_slider(
        "回放帧", min=0, max=max(1, len(session.records) - 1), step=1, initial_value=0
      )
      self.timeline.on_update(
        lambda e: (
          self.commands.put(("seek", int(e.target.value)))
          if e.client_id is not None
          else None
        )
      )
    self.session.paused = True

    @self.server.on_client_connect
    def connected(client):
      client.camera.position = (3.6, -4.2, 2.4)
      client.camera.look_at = (0.7, 0.5, 0.85)
      client.camera.up_direction = (0.0, 0.0, 1.0)
      client._websock_connection.queue_message(
        _messages.RunJavascriptMessage(
          source=Path(__file__).with_name("keyboard.js").read_text()
        )
      )

    self.update()

  def drain(self, output):
    self.single_step = False
    while not self.commands.empty():
      command, value = self.commands.get_nowait()
      try:
        if command == "mode":
          self.session.set_mode(value)
          self.session.paused = True
          self.pause_handle.value = True
        elif command == "pause":
          self.session.paused = bool(value)
          self.pause_handle.value = self.session.paused
        elif command == "toggle_pause":
          self.session.paused = not self.session.paused
          self.pause_handle.value = self.session.paused
        elif command == "reset":
          self.session.reset()
          self.session.paused = True
          self.pause_handle.value = True
        elif command in ("motion", "adjacent_motion", "motion_seek"):
          if command == "motion_seek":
            self.session.seek_motion(value)
          else:
            path = (
              self.session.motion_catalog[value]
              if command == "motion"
              else self.session.adjacent_motion(value)
            )
            if path != self.session.motion:
              self.session.set_motion(path)
          self.session.paused = True
          self.pause_handle.value = True
        elif command == "loop":
          self.session.tracker_loop = bool(value)
        elif command == "feed":
          self.session.feed()
        elif command == "step":
          self.single_step = True
        elif command == "scenario":
          self.session.scenario = value
          self.session.ball.cfg.freeze_on_done = value != "continuous"
        elif command == "speed":
          self.session.speed = value
        elif command == "rate":
          self.rate = value
        elif command == "latent":
          self.session.latent[value[0]] = value[1]
        elif command == "seek":
          self.session.seek(value)
        elif command == "save":
          if self.session.mode == "replay":
            self.session.save(output)
          else:
            self.session.save(output, completed=False)
          self.message.content = f"已保存：`{output}`"
        elif command == "stop":
          self.running = False
        self.message.content = "" if command != "save" else self.message.content
      except (ValueError, FileNotFoundError) as error:
        self.message.content = f"操作未执行：{error}"

  def update(self, fraction=1.0):
    s = self.session
    self.frames.capture(s)
    display = self.frames.sample(fraction)
    if s.info.get("control_fault"):
      self.message.content = (
        "**策略数值失败，当前仅保持最后有效关节目标，继续记录物理状态。R 可重置。**\n\n"
        + s.info["control_fault"]
      )
    self.speed_handle.disabled = s.mode not in ("tracker", "mvae", "mvae_tracker")
    if self.motion_picker is not None:
      enabled = s.mode == "tracker"
      self.motion_picker.disabled = not enabled
      for handle in self.motion_buttons:
        handle.disabled = not enabled
      assert self.motion_time is not None
      self.motion_time.disabled = not enabled
      self.motion_time.max = max(0.001, s.controller.reference.duration)
      self.motion_time.value = s.controller.reference_time
      self.motion_picker.value = next(
        name for name, path in s.motion_catalog.items() if path == s.motion
      )
    self.pause_handle.value = s.paused
    if getattr(s, "record_limit_reached", False):
      self.message.content = "已达到记录帧数上限并暂停。请保存记录，重新启动一次测试。"
      self.pause_handle.value = True
    self.scene.update_from_mjdata(self.frames.data)
    self.ball.position = display["ball"]
    self.ball.visible = s.info.get("ball_active", s.active)
    self.path.visible = bool(s.ball_obs[0, -2])
    points = display["future"]
    self.path.points = np.stack((points[:-1], points[1:]), 1)
    if s.info:
      r = s.info
      offset = np.array([0.0, 1.4, 0.0]) if self.layout.value == "并排" else np.zeros(3)
      self.reference_robot.update(
        display["reference_qpos"], offset, self.show_reference.value, self.opacity.value
      )
      self.actual_label.position = display["qpos"][:3] + [0.0, 0, 0.9]
      self.actual_label.text = "实际执行" if not r["kinematic"] else "运动学预览"
      self.actual.position = display["racket"]
      self.reference.position = display["reference_racket"] + offset
      p = display["racket"]
      self.normal.points = np.array([[p, p + 0.15 * display["normal"]]])
      reference = s.controller.current
      positions = display["positions"]
      self.skeleton.points = np.array(
        [
          [positions[reference.fk.parents[i] - 1] + offset, positions[i] + offset]
          for i in range(1, len(positions))
        ]
      )
      self.skeleton.visible = self.ghost.value
      self.reference.visible = self.show_reference.value or self.ghost.value
      if self.timeline is not None:
        self.timeline.value = s.index
      now = time.monotonic()
      if now - self.last_status_update < 0.1:
        return
      self.last_status_update = now
      self.chart_counter += 1
      if self.chart_counter % 2 == 0:
        end = s.index + 1 if s.mode == "replay" else s.records.total
        rows = (
          s.records[max(0, end - 250) : end] if s.mode == "replay" else s.records[-250:]
        )
        self.chart.data = (
          np.arange(end - len(rows), end, dtype=float),
          np.array([row["racket_error_m"] * 100 for row in rows]),
          np.array([row["joint_rms_rad"] for row in rows]),
        )
      motion_status = ""
      if s.mode == "tracker":
        assert self.motion_picker is not None
        motion_status = (
          f"\n\nMotion：{self.motion_picker.value}"
          f"\n\n进度 {s.controller.reference_time:.3f} / "
          f"{s.controller.reference.duration:.3f}s"
          + (" · 播放完毕" if s.tracker_at_end else "")
        )
      timing_status = (
        f"\n\n物理 {1 / s.world.dt:.0f} Hz · 控制 {1 / s.world.control_dt:.0f} Hz"
        f"\n\n播放 {self.rate:.2f}× · 发送 {self.playback_stats['fps']:.0f} FPS"
        f" · 实测仿真 / 墙钟 {self.playback_stats['realtime_factor']:.2f}×"
      )
      count = len(s.records) if s.mode == "replay" else s.records.total
      lifecycle = (
        "录像回放"
        if s.mode == "replay"
        else "连续：不自动重置或因摔倒暂停"
        if s.scenario == "continuous"
        else "独立单球：每球重置"
      )
      self.status.content = f"**模式：{s.mode} / 当前策略：{r['policy']}**{motion_status}{timing_status}\n\n{lifecycle}\n\n时间 {s.elapsed:.2f}s · alpha {r['alpha']:.2f}\n\n拍心误差 {r['racket_error_m'] * 100:.1f}cm · 关节 RMS {r['joint_rms_rad']:.3f}rad\n\n击球 {r['hit']} / 过网 {r['net_clear']} / 落台 {r['success']}\n\n摔倒 {r['fallen']} · 已记录 {count} 帧 / {len(s.trials)} 球\n\n{'运动学生成预览（不评判物理稳定性）' if r['kinematic'] else '原生 CPU MuJoCo 物理执行'}"

  def close(self):
    self.server.stop()

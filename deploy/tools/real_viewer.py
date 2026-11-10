"""Live real-robot telemetry and exact recorded-event replay; never sends commands."""

import argparse
import bisect
import copy
import hashlib
import inspect
import json
import queue
import signal
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

from deploy.io.diagnostics import json_default


def finite_vector(value, length):
  try:
    array = np.asarray(value, dtype=float)
    return array.shape == (length,) and bool(np.isfinite(array).all())
  except (TypeError, ValueError):
    return False


class DisplayState:
  """Causal sample-and-hold, with separate timestamps for robot and ball."""

  def __init__(self):
    self.robot = {}
    self.robot_time = self.ball_time = self.command_time = None
    self.ball = self.command = self.prediction = self.reference = None
    self.segment = None
    self.trail = deque(maxlen=800)
    self.last_source_stamp = None
    self.now = 0.0

  def apply(self, event):
    t, kind, data = event["time"], event["kind"], event["data"]
    self.now = max(self.now, t)
    if kind == "state":
      self.robot, self.robot_time = data, t
    elif kind == "mocap":
      if data.get("frame") != "hope_table":
        return
      if data.get("kind") == "ball" and finite_vector(data.get("position"), 3):
        stamp = data.get("stamp")
        if stamp == self.last_source_stamp:
          return
        if self.last_source_stamp is not None and (
          stamp < self.last_source_stamp or stamp - self.last_source_stamp > 0.1
        ):
          self.trail.clear()
          self.prediction = self.reference = None
        self.last_source_stamp = stamp
        self.ball = data["position"]
        self.ball_time = data.get("sample_monotonic", t)
        self.trail.append((t, self.ball))
        while self.trail and t - self.trail[0][0] > 5:
          self.trail.popleft()
      elif data.get("kind") == "racket_command":
        self.command, self.command_time = data, t
    elif kind == "prediction":
      if self.segment != data["segment"]:
        self.reference = data
        self.segment = data["segment"]
      self.prediction = data
    elif kind == "prediction_reset":
      self.prediction = self.reference = None
      self.segment = None
      if data.get("type") == "reset":
        self.trail.clear()
        if self.ball is not None:
          self.trail.append((self.ball_time, self.ball))

  def robot_valid(self, now):
    row = self.robot
    age = now - self.robot_time if self.robot_time is not None else float("inf")
    return (
      self.robot_time is not None
      and 0 <= now - self.robot_time <= 0.2
      and finite_vector(row.get("q"), 29)
      and finite_vector(row.get("root_pos"), 3)
      and finite_vector(row.get("root_quat"), 4)
      and np.linalg.norm(row["root_quat"]) > 1e-8
      and row.get("root_age") is not None
      and 0 <= row["root_age"] + age <= 0.2
      and row.get("dds_state_age") is not None
      and 0 <= row["dds_state_age"] + age <= 0.2
    )


class Scene:
  def __init__(self, directory, *, host, port, replay=False):
    import viser

    self.replay = replay
    self.server = viser.ViserServer(host=host, port=port, label="实机观测与回放")
    if self.server.get_port() != port:
      self.server.stop()
      raise RuntimeError(f"Port {port} is occupied")
    self.server.scene.set_up_direction("+z")
    self.server.gui.add_markdown(
      "**实机观测回放**" if replay else "**实机只读观测 · 后台记录**"
    )
    self.server.gui.add_markdown(
      "绿色：动捕轨迹 · 橙色：辅助预测 · 蓝色：本段首次预测 · 紫色：收到的规划目标。\n\n"
      "辅助预测复用 planner 物理模型，不包含球拍/球网碰撞；不是控制器指令。"
    )
    self.status = self.server.gui.add_markdown("等待状态…")
    self.joints = self.server.gui.add_markdown("")
    self.show_joints = self.server.gui.add_checkbox(
      "显示实测关节角（度）", initial_value=False
    )
    self.paused = self.server.gui.add_checkbox(
      "暂停回放", initial_value=True, visible=replay
    )
    self.speed = self.server.gui.add_slider(
      "回放倍速", min=0.1, max=4, step=0.1, initial_value=1, visible=replay
    )
    self.restart = False
    self.seek_to = None
    if replay:
      self.progress = self.server.gui.add_slider(
        "回放进度（秒）", min=0.0, max=1.0, step=0.01, initial_value=0.0
      )
      self.timeline = self.server.gui.add_markdown("正在读取日志…")
      self.load_error = self.server.gui.add_markdown("")

      @self.progress.on_update
      def seek(event):
        # Server-side progress updates must not become user seek requests.
        if event.client is not None:
          self.seek_to = float(event.target.value)
          self.paused.value = True

      self.server.gui.add_button("从头回放").on_click(
        lambda _: setattr(self, "restart", True)
      )
    self.log_path = self.server.gui.add_markdown("")
    try:
      self.load_recording(directory)
    except Exception:
      self.server.stop()
      raise

    @self.server.on_client_connect
    def connect(client):
      client.camera.position = np.array([-2.5, -4, 2.5]) - self.offset
      client.camera.look_at = np.array([0.7, -0.76, 0.25]) - self.offset
      client.camera.up_direction = (0, 0, 1)

  def load_recording(self, directory):
    """Switch models without restarting the web server or its GUI controls."""
    import mujoco
    from mjviser import ViserMujocoScene

    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    model_path = directory / "scene.mjb"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta["model_sha256"]:
      raise ValueError("Recorded scene checksum mismatch")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = [
      model.jnt_qposadr[model.joint("robot/" + n).id] for n in meta["joint_names"]
    ]
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
      raise ValueError("Expected exactly one robot free joint")
    self.server.scene.reset()
    self.directory, self.meta, self.model, self.data = directory, meta, model, data
    self.addresses = addresses
    self.offset = np.asarray(meta["world_translation"])
    self.root = model.jnt_qposadr[free[0]]
    self.geometry = ViserMujocoScene(self.server, model, num_envs=1)
    self.geometry.camera_tracking_enabled = False
    self.lines = {}
    style = (
      {"thickness": 3.0, "thickness_units": "screen"}
      if "thickness"
      in inspect.signature(self.server.scene.add_line_segments).parameters
      else {"line_width": 3.0}
    )
    for name, color in dict(
      measured=(26, 255, 90), latest=(255, 140, 13), reference=(89, 166, 255)
    ).items():
      self.lines[name] = self.server.scene.add_line_segments(
        "/live/" + name,
        points=np.zeros((1, 2, 3), np.float32),
        colors=color,
        visible=False,
        **style,
      )
    self.ball = self.server.scene.add_icosphere(
      "/live/ball", radius=0.02, color=(255, 255, 255), visible=False
    )
    self.command = self.server.scene.add_icosphere(
      "/live/command", radius=0.035, color=(220, 50, 230), visible=False
    )
    self.log_path.content = (
      "只读日志回放，不连接机器人。\n\n"
      if self.replay
      else "关闭网页不会停止机器人或录制。结束部署后，可独立回放此目录。\n\n"
    ) + f"日志：`{self.directory}`"

  def update(self, state, now):
    import mujoco

    valid = state.robot_valid(now)
    if valid:
      self.data.qpos[self.addresses] = state.robot["q"]
      self.data.qpos[self.root : self.root + 3] = (
        np.asarray(state.robot["root_pos"]) - self.offset
      )
      quat = np.asarray(state.robot["root_quat"])
      self.data.qpos[self.root + 3 : self.root + 7] = quat / np.linalg.norm(quat)
    else:
      # Hide unavailable measured robot below the scene, keeping the table.
      self.data.qpos[self.root : self.root + 3] = [0, 0, -100]
    mujoco.mj_kinematics(self.model, self.data)
    self.geometry.update_from_mjdata(self.data)
    fresh_ball = state.ball_time is not None and 0 <= now - state.ball_time <= 0.1
    self.ball.visible = fresh_ball
    if fresh_ball:
      self.ball.position = np.asarray(state.ball) - self.offset
    for name, points in dict(
      measured=[p for t, p in state.trail if now - t <= 5],
      latest=state.prediction["points"] if fresh_ball and state.prediction else [],
      reference=state.reference["points"] if state.reference else [],
    ).items():
      line = self.lines[name]
      line.visible = len(points) >= 2
      if line.visible:
        p = np.asarray(points, dtype=np.float32) - self.offset
        line.points = np.stack([p[:-1], p[1:]], axis=1).astype(np.float32)
        # Preserve uniform RGB; changing color-array shape breaks Viser shaders.
    cmd = state.command
    self.command.visible = bool(
      cmd
      and cmd.get("valid")
      and state.command_time is not None
      and 0 <= now - state.command_time <= 0.15
      and finite_vector(cmd.get("position"), 3)
    )
    if self.command.visible:
      self.command.position = np.asarray(cmd["position"]) - self.offset
    age = "—" if state.robot_time is None else f"{now - state.robot_time:.3f}s"
    ball_age = "—" if state.ball_time is None else f"{now - state.ball_time:.3f}s"
    pos = state.robot.get("root_pos")
    self.status.content = (
      f"状态：**{state.robot.get('state', '等待')}** · {state.robot.get('phase', '')}\n\n"
      f"机器人：{'有效' if valid else '缺失/过期，已隐藏'} · 帧延迟 {age}\n\n"
      f"骨盆（球台坐标，m）：{pos}\n\n球观测延迟：{ball_age}"
    )
    self.joints.content = (
      "|关节|实测 °|\n|---|---:|\n"
      + "\n".join(
        f"|{n}|{np.degrees(q):.1f}|"
        for n, q in zip(self.meta["joint_names"], state.robot["q"], strict=True)
      )
      if valid and self.show_joints.value
      else ""
    )

  def close(self):
    self.server.stop()


def live_worker(directory, events, stop, ready, host, port, headless):
  # The parent owns Ctrl-C and stops this process after robot IO has closed.
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  directory = Path(directory)
  scene = None
  counts = dict(events=0, predictions=0)
  try:
    from deploy.hope_g1_hitter.planner_source import use_sibling_planner

    use_sibling_planner()
    from hope_planner_g1.profiles import load_profile
    from hope_planner_g1.trajectory import TrajectoryMonitor

    meta = json.loads((directory / "metadata.json").read_text())
    mode = meta["physics"]["runtime_mode"]
    profile = load_profile(
      mode, directory / "calibration.yaml" if mode == "real" else None
    )
    if profile.source_sha256 != meta["physics"]["source_sha256"]:
      raise ValueError("Viewer physics differs from deployment physics")
    state = DisplayState()
    with (directory / "events.jsonl").open("w", buffering=1) as stream:
      event_time = time.monotonic()

      def record(event):
        stream.write(json.dumps(event, default=json_default, allow_nan=False) + "\n")
        state.apply(event)
        counts["events"] += 1

      def prediction(row):
        if row["type"] in ("prediction", "reset", "observed_bounce"):
          kind = "prediction" if row["type"] == "prediction" else "prediction_reset"
          record(dict(kind=kind, time=event_time, data=row))
          counts["predictions"] += kind == "prediction"

      monitor = TrajectoryMonitor(
        profile, prediction, prediction_hz=15, frame="hope_table"
      )
      if not headless:
        scene = Scene(directory, host=host, port=port)
      ready.set()
      last_draw = 0.0
      while True:
        try:
          event = events.get(timeout=0.01)
        except queue.Empty:
          if stop.is_set():
            break
          event = None
        if event is not None:
          event_time = event["time"]
          record(event)
          data = event["data"]
          if event["kind"] == "mocap" and data.get("kind") == "ball":
            monitor.observe(
              data["stamp"],
              data["position"],
              frame=data["frame"],
              arrival_stamp=data["stamp"]
              + (event_time - data.get("sample_monotonic", event_time)),
            )
        now = time.monotonic()
        if scene is not None and now - last_draw >= 1 / 30:
          try:
            scene.update(state, now)
          except Exception:
            (directory / "display_error.txt").write_text(traceback.format_exc())
            scene.close()
            scene = None  # Recording and prediction continue after display failure.
          last_draw = now
      counts["trajectory"] = monitor.summary()
  except Exception:
    (directory / "worker_error.txt").write_text(traceback.format_exc())
    raise
  finally:
    if scene is not None:
      scene.close()
    (directory / "summary.json").write_text(json.dumps(counts, indent=2))


def read_events(path):
  with Path(path).open() as stream:
    for line in stream:
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        # A hard-killed process can leave one incomplete final record.
        if stream.read().strip():
          raise
        return


class ReplayLog:
  """Sparse state checkpoints for causal seeks without keeping every event in RAM.

  Each open snapshots the file size. An actively growing recording can be
  reopened to include newer events, without moving the timeline while dragging.
  """

  def __init__(self, directory, checkpoint_seconds=5.0):
    path = Path(directory) / "events.jsonl"
    self.end_offset = path.stat().st_size
    self.checkpoints, self.times = [], []
    self.start = self.end = None
    state = DisplayState()
    with path.open("rb") as stream:
      while (event := self._read(stream)) is not None:
        t = float(event["time"])
        if not np.isfinite(t):
          raise ValueError("Nonfinite event timestamp")
        if self.start is None:
          self.start = self.end = t
          self.times.append(t)
          self.checkpoints.append((0, DisplayState()))
        # Retain recorded order, including occasional late sensor events.
        self.end = max(self.end, t)
        state.apply(event)
        if self.end - self.times[-1] >= checkpoint_seconds:
          self.times.append(self.end)
          self.checkpoints.append((stream.tell(), copy.deepcopy(state)))
    if self.start is None:
      raise ValueError(f"No complete events in {path}")
    self.duration = self.end - self.start
    self.stream = path.open("rb")
    self.seek(0)

  def _read(self, stream):
    remaining = self.end_offset - stream.tell()
    if remaining <= 0:
      return None
    line = stream.readline(remaining)
    try:
      return json.loads(line)
    except json.JSONDecodeError:
      if stream.read(max(0, self.end_offset - stream.tell())).strip():
        raise
      return None  # Ignore an incomplete final line, as read_events does.

  def seek(self, seconds):
    self.position = float(np.clip(seconds, 0, self.duration))
    index = bisect.bisect_right(self.times, self.start + self.position) - 1
    offset, state = self.checkpoints[index]
    self.state = copy.deepcopy(state)
    self.stream.seek(offset)
    self.pending = self._read(self.stream)
    self.advance(self.position)

  def advance(self, seconds):
    seconds = float(np.clip(seconds, 0, self.duration))
    if seconds < self.position:
      self.seek(seconds)
      return
    self.position = seconds
    while self.pending is not None and self.pending["time"] <= self.now:
      self.state.apply(self.pending)
      self.pending = self._read(self.stream)

  @property
  def now(self):
    return self.start + self.position

  def close(self):
    self.stream.close()


def find_recordings(root):
  """One selectable row per run containing a complete visualization bundle."""
  root = Path(root).expanduser().resolve()
  if not root.is_dir():
    raise ValueError(f"Recording root is not a directory: {root}")
  recordings = {}
  for run in sorted(root.iterdir(), reverse=True):
    directory = run / "visualization"
    if all(
      (directory / name).is_file()
      for name in ("events.jsonl", "metadata.json", "scene.mjb")
    ):
      recordings[run.name] = directory
  return recordings


def replay(
  directory=None, *, replay_dir=None, host, port, autoplay=False, duration=None
):
  """Recorded playback with seeks and optional run selection; no robot IO."""
  recordings = find_recordings(replay_dir) if replay_dir is not None else None
  if recordings is not None:
    if not recordings:
      raise ValueError(f"No visualization recordings found under {replay_dir}")
    selected = next(iter(recordings))
    directory = recordings[selected]
  log = ReplayLog(Path(directory).expanduser())
  try:
    scene = Scene(directory, host=host, port=port, replay=True)
  except Exception:
    log.close()
    raise
  scene.paused.value = not autoplay
  selection = None
  refresh = False
  if recordings is not None:
    selection = scene.server.gui.add_dropdown(
      "运行记录（最新在前）",
      options=list(recordings),
      initial_value=selected,
      order=-10,
    )

    def request_refresh(_):
      nonlocal refresh
      refresh = True

    scene.server.gui.add_button("刷新记录列表 / 重新读取当前日志", order=-9).on_click(
      request_refresh
    )

  def reset_timeline():
    scene.progress.max = max(0.01, log.duration)
    scene.progress.value = 0.0
    scene.progress.disabled = log.duration == 0
    scene.seek_to = None
    scene.restart = False

  reset_timeline()
  started = previous = time.monotonic()
  try:
    while duration is None or time.monotonic() - started < duration:
      now = time.monotonic()
      if selection is not None and (refresh or selection.value != selected):
        requested = selection.value
        try:
          if refresh:
            found = find_recordings(replay_dir)
            if not found:
              raise ValueError("没有可用的 visualization 记录")
            recordings = found
            selection.options = list(recordings)
            requested = requested if requested in recordings else next(iter(recordings))
          new_log = ReplayLog(recordings[requested])
          try:
            scene.load_recording(recordings[requested])
          except Exception:
            new_log.close()
            raise
          log.close()
          log = new_log
          selected = requested
          selection.value = selected
          scene.paused.value = True
          scene.load_error.content = ""
          reset_timeline()
          previous = time.monotonic()
        except Exception as exc:
          scene.load_error.content = f"读取记录失败：{exc}"
          if selected not in selection.options:
            selection.options = [selected, *selection.options]
          selection.value = selected
          scene.paused.value = True
        refresh = False
        now = time.monotonic()
      if scene.restart:
        log.seek(0)
        scene.restart = False
      if scene.seek_to is not None:
        seconds, scene.seek_to = scene.seek_to, None
        log.seek(seconds)
      if not scene.paused.value:
        log.advance(log.position + (now - previous) * float(scene.speed.value))
        if log.position >= log.duration:
          scene.paused.value = True
      previous = now
      scene.progress.value = log.position
      scene.timeline.content = f"**{log.position:.2f} / {log.duration:.2f} 秒** · 拖动进度条后暂停，取消暂停继续播放。"
      scene.update(log.state, log.now)
      time.sleep(1 / 30)
  finally:
    log.close()
    scene.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--replay", type=Path, help="Recorded visualization directory")
  source.add_argument(
    "--replay-dir",
    type=Path,
    help="Parent recording directory; choose a run in the web viewer",
  )
  parser.add_argument("--web-host", default="127.0.0.1")
  parser.add_argument("--web-port", type=int, default=8081)
  parser.add_argument("--autoplay", action="store_true")
  parser.add_argument("--duration", type=float)
  args = parser.parse_args()
  if not 1 <= args.web_port <= 65535 or (
    args.duration is not None and args.duration <= 0
  ):
    parser.error("Invalid port or duration")
  try:
    replay(
      args.replay,
      replay_dir=args.replay_dir,
      host=args.web_host,
      port=args.web_port,
      autoplay=args.autoplay,
      duration=args.duration,
    )
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()

"""Recorded-state MuJoCo viewer. No controller, ROS, DDS or physics rollout."""

import argparse
import bisect
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from deploy.backends.playback import HELP, PlaybackControl


def json_rows(path):
  """Permit a final incomplete write from a recording still being copied."""
  with path.open() as stream:
    for number, line in enumerate(stream, 1):
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        if stream.read().strip():
          raise ValueError(f"Malformed JSON in {path}:{number}") from None
        print(f"Ignoring incomplete final row: {path}:{number}")


def vector(value, size):
  try:
    result = np.asarray(value, dtype=float)
    return result if result.shape == (size,) and np.isfinite(result).all() else None
  except (TypeError, ValueError):
    return None


def received_command(row):
  """Home goals and unaccepted planner predictions are not received commands."""
  command = row.get("raw_racket_command")
  if (row.get("state") == "EXECUTE" or row.get("phase") == "SWING") and isinstance(
    command, dict
  ):
    return command
  return None


class Recording:
  def __init__(
    self,
    directory,
    *,
    rally_config=None,
    assume_hope_table_coordinates=False,
    command_view="locked",
  ):
    if command_view not in ("locked", "received"):
      raise ValueError("command_view must be locked or received")
    self.command_view = command_view
    self.directory = Path(directory).resolve()
    self.summary = json.loads((self.directory / "summary.json").read_text())
    self.source_world = self.summary.get("control_world_frame")
    if self.source_world != "hope_table":
      if self.source_world != "world" or not assume_hope_table_coordinates:
        raise ValueError(
          "This log is not labeled hope_table. For a verified identity-frame "
          "legacy log, pass --assume-hope-table-coordinates explicitly."
        )
    fallback = yaml.safe_load(Path(rally_config).read_text()) if rally_config else {}
    self.ball_config = self.summary.get("rally_runtime_config", {}).get("ball")
    if self.ball_config is None:
      self.ball_config = fallback.get("ball")
    if self.ball_config is None:
      raise ValueError(
        "No table/ball geometry in summary; provide --rally-config with a ball section"
      )
    self.offset = vector(self.summary.get("simulation_world_translation"), 3)
    if self.offset is None:
      self.offset = vector(
        fallback.get("world", {}).get("mujoco_to_world_translation"), 3
      )
    if self.offset is None:
      raise ValueError(
        "No MuJoCo/world translation in summary; provide --rally-config with "
        "world.mujoco_to_world_translation"
      )
    self.names = self.summary["robot_command_contract"]["joint_names"]
    if len(set(self.names)) != 29:
      raise ValueError("Expected a recorded contract with 29 unique G1 joints")
    self.rows, self.times, self.events = [], [], []
    self.skipped = 0
    keys = (
      "time",
      "q",
      "root_pos",
      "root_quat",
      "state",
      "phase",
      "motion_index",
      "goal_position",
      "goal_normal",
      "goal_velocity",
      "raw_racket_command",
      "time_to_strike",
      "racket_pos",
      "racket_normal",
      "ball_position",
      "ball_age",
      "root_age",
    )
    for row in json_rows(self.directory / "trace.jsonl"):
      stamp = row.get("time")
      if (
        not isinstance(stamp, (int, float))
        or not math.isfinite(stamp)
        or any(
          vector(row.get(k), n) is None
          for k, n in (("q", 29), ("root_pos", 3), ("root_quat", 4))
        )
        or np.linalg.norm(row["root_quat"]) < 1e-8
      ):
        self.skipped += 1
        continue
      if self.times and stamp < self.times[-1]:
        raise ValueError("Trace time moves backwards; select one recording folder")
      if not self.rows:
        self.origin = stamp
      t = stamp - self.origin
      saved = {key: row.get(key) for key in keys}
      if not self.rows or (saved["state"], saved["phase"], saved["motion_index"]) != (
        self.rows[-1]["state"],
        self.rows[-1]["phase"],
        self.rows[-1]["motion_index"],
      ):
        self.events.append(t)
      self.times.append(t)
      self.rows.append(saved)
    if not self.rows:
      raise ValueError("No usable measured robot frames in trace.jsonl")
    self.duration = self.times[-1]
    self.locked_commands, self.locked_times = [], []
    events = self.summary.get("events", [])
    for event in events:
      raw = event.get("raw_command")
      if event.get("event") != "motion_selected" or not isinstance(raw, dict):
        continue
      if vector(raw.get("position"), 3) is None:
        continue
      start = event["time"] - self.origin
      end = next(
        (
          item["time"] - self.origin
          for item in events
          if item["time"] > event["time"]
          and item.get("event") in ("recovered", "motion_selected")
        ),
        self.duration,
      )
      self.locked_times.append(start)
      self.locked_commands.append(
        dict(
          raw,
          received=raw.get("received_monotonic", event["time"]) - self.origin,
          deadline=event["deadline"] - self.origin,
          end=end,
          source="LOCKED execution command",
          motion_index=event["motion_index"],
        )
      )
    self.ball_times, self.ball_positions = [], []
    self.command_times, self.commands = [], []
    self._crossings = {}
    self._encounters = {}
    packets = self.directory / "mocap_packets.jsonl"
    if packets.is_file():
      for row in json_rows(packets):
        packet = row.get("packet", {})
        p = vector(packet.get("position"), 3)
        arrival = row.get("arrival_time")
        if packet.get("kind") == "racket_command" and arrival is not None:
          if packet.get("frame") != self.source_world:
            raise ValueError("Command packet frame differs from recorded robot world")
          self.command_times.append(float(arrival) - self.origin)
          self.commands.append(packet)
          continue
        if packet.get("kind") != "ball" or p is None or arrival is None:
          continue
        if packet.get("frame") != self.source_world:
          raise ValueError("Ball packet frame differs from recorded robot world")
        self.ball_times.append(float(arrival) - self.origin)
        self.ball_positions.append(p)
    if self.ball_times:
      order = np.argsort(self.ball_times, kind="stable")
      self.ball_times = np.asarray(self.ball_times)[order].tolist()
      self.ball_positions = np.asarray(self.ball_positions)[order]
    if self.command_times:
      order = np.argsort(self.command_times, kind="stable")
      self.commands = [self.commands[i] for i in order]
      self.command_times = np.asarray(self.command_times)[order].tolist()
    measured = [
      (t, vector(r.get("racket_pos"), 3), vector(r.get("racket_normal"), 3))
      for t, r in zip(self.times, self.rows, strict=True)
    ]
    measured = [(t, p, n) for t, p, n in measured if p is not None and n is not None]
    self.racket_times = np.array([t for t, _, _ in measured])
    self.racket_positions = np.array([p for _, p, _ in measured])
    self.racket_normals = np.array([n for _, _, n in measured])

  def racket_at(self, seconds, *, available=math.inf):
    """Measured kinematics interpolated only across <=50ms recorded intervals."""
    i = int(np.searchsorted(self.racket_times, seconds))
    if i < len(self.racket_times) and self.racket_times[i] > available:
      return None
    if i < len(self.racket_times) and abs(self.racket_times[i] - seconds) < 1e-9:
      return self.racket_positions[i], self.racket_normals[i]
    if i == 0 or i == len(self.racket_times):
      return None
    a, b = self.racket_times[i - 1 : i + 1]
    if not 0 < b - a <= 0.05:
      return None
    f = (seconds - a) / (b - a)
    p = self.racket_positions[i - 1] * (1 - f) + self.racket_positions[i] * f
    n = self.racket_normals[i - 1] * (1 - f) + self.racket_normals[i] * f
    if np.linalg.norm(n) < 1e-8:
      return None
    return p, n / np.linalg.norm(n)

  def encounter(self, seconds, command):
    """Estimated paddle deflection, otherwise a measured plane crossing.

    A reversal alone is insufficient: require proximity to the reconstructed
    finite paddle. 3cm geometry tolerance accounts for uncertain reconstruction;
    this is explicitly a contact estimate, never a confirmed collision label.
    """
    goal = np.asarray(command["position"])
    since = command["received"]
    deadline = command.get("deadline")
    key = (since, deadline, tuple(goal))
    if key not in self._encounters:
      hit = None
      if deadline is not None:
        lo, hi = max(since, deadline - 0.25), deadline + 0.25
        bt = np.asarray(self.ball_times)
        bp = self.ball_positions
        candidates = range(
          max(1, np.searchsorted(bt, lo)), min(len(bt) - 1, np.searchsorted(bt, hi))
        )
        radius = self.ball_config.get("racket_radius", 0.075)
        ball_radius = self.ball_config.get("ball_radius", 0.02)
        for i in candidates:
          t = bt[i]
          if not bp[i - 1][0] > bp[i][0] <= bp[i + 1][0]:
            continue
          before = np.flatnonzero((bt >= t - 0.06) & (bt <= t - 0.02))
          after = np.flatnonzero((bt >= t + 0.02) & (bt <= t + 0.07))
          if len(before) < 3 or len(after) < 3:
            continue
          segment = bt[before[0] : after[-1] + 1]
          if np.max(np.diff(segment)) > 0.05:
            continue

          def vx(ids, t=t):
            return np.linalg.lstsq(
              np.c_[bt[ids] - t, np.ones(len(ids))], bp[ids, 0], rcond=None
            )[0][0]

          if vx(before) >= -0.5 or vx(after) <= 0.5:
            continue
          # Require the reversal to coincide with the paddle encounter. A later
          # body/table deflection must not replace a missed x-hit crossing.
          nearby = []
          for j in range(
            max(0, np.searchsorted(bt, lo)),
            min(len(bt), np.searchsorted(bt, min(hi, bt[after[-1]]), side="right")),
          ):
            measured = self.racket_at(bt[j])
            if measured is None:
              continue
            p, n = measured
            delta = bp[j] - p
            axial = abs(float(delta @ n))
            radial = np.sqrt(max(0, delta @ delta - axial**2))
            distance = np.hypot(axial, max(0, radial - radius))
            nearby.append((distance, bt[j]))
          measured = self.racket_at(t)
          if not nearby or measured is None:
            continue
          distance, near_time = min(nearby)
          p, _ = measured
          if distance > ball_radius + 0.03 or abs(t - near_time) > 0.03:
            continue
          ready_index = min(
            len(self.racket_times) - 1,
            np.searchsorted(self.racket_times, bt[after[-1]]),
          )
          hit = dict(
            kind="estimated_hit",
            position=bp[i],
            time=t,
            available=max(bt[after[-1]], self.racket_times[ready_index]),
            racket_position=p,
          )
          break
      self._encounters[key] = hit
    hit = self._encounters[key]
    if hit is not None and hit["available"] <= seconds:
      return hit
    crossing = self.crossing(seconds, goal[0], since=since)
    if crossing is None:
      return None
    measured = self.racket_at(crossing["time"], available=seconds)
    return dict(
      crossing,
      kind="crossing",
      racket_position=measured[0] if measured is not None else None,
    )

  def command(self, seconds):
    """Default: immutable execution goal from selection through recovery.

    The optional received view follows live UDP candidates, not robot targets.
    Legacy logs without selection events fall back to accepted trace goals.
    """
    if self.command_view == "locked" and self.locked_times:
      i = bisect.bisect_right(self.locked_times, seconds) - 1
      if i < 0 or seconds >= self.locked_commands[i]["end"]:
        return None
      return self.locked_commands[i]
    if self.command_view == "received" and self.command_times:
      i = bisect.bisect_right(self.command_times, seconds) - 1
      if i < 0:
        return None
      packet, received = self.commands[i], self.command_times[i]
      p = vector(packet.get("position"), 3)
      tts = packet.get("time_to_strike")
      if (
        not packet.get("valid")
        or p is None
        or not isinstance(tts, (float, int))
        or not math.isfinite(tts)
        or tts <= 0
      ):
        return None
      published = packet.get("bridge_monotonic", received + self.origin)
      deadline = float(published) - self.origin + tts
      if seconds > deadline + 2:
        return None
      return dict(packet, received=received, deadline=deadline, source="received UDP")
    row = self.rows[self.index(seconds)]
    raw = received_command(row)
    if raw is None:
      return None
    p = vector(raw.get("position"), 3)
    if p is None:
      p = vector(row.get("goal_position"), 3)
    if p is None:
      return None
    index = self.index(seconds)
    while index > 0 and received_command(self.rows[index - 1]) == raw:
      index -= 1
    return dict(
      position=p,
      normal=raw.get("normal", row.get("goal_normal")),
      velocity=raw.get("velocity", row.get("goal_velocity")),
      received=self.times[index],
      source="accepted trace (no receipt packets)",
    )

  def crossing(self, seconds, plane_x, *, since=-math.inf):
    """Incoming +X -> -X crossing, measured samples only; no gap >50ms.

    available is the second packet arrival: never reveal a crossing before its
    bracketing measurement has actually arrived. Time is on the arrival clock.
    """
    x = float(plane_x)
    if x not in self._crossings:
      found = []
      for i in range(1, len(self.ball_times)):
        a, b = self.ball_times[i - 1 : i + 1]
        p, q = self.ball_positions[i - 1 : i + 1]
        if 0 < b - a <= 0.05 and p[0] > x >= q[0]:
          fraction = (p[0] - x) / (p[0] - q[0])
          found.append(
            dict(
              time=a + fraction * (b - a), available=b, position=p + fraction * (q - p)
            )
          )
      self._crossings[x] = found
    for item in reversed(self._crossings[x]):
      if item["available"] <= seconds and item["time"] >= since:
        return item
    return None

  def index(self, seconds):
    return max(0, bisect.bisect_right(self.times, seconds) - 1)

  def ball(self, seconds, max_age=0.1):
    i = bisect.bisect_right(self.ball_times, seconds) - 1
    if i < 0 or seconds - self.ball_times[i] > max_age:
      return None
    return self.ball_positions[i]

  def describe(self):
    return dict(
      recording=str(self.directory),
      seconds=self.duration,
      measured_robot_frames=len(self.rows),
      skipped_frames=self.skipped,
      raw_ball_samples=len(self.ball_times),
      received_planner_packets=len(self.command_times),
      locked_execution_commands=len(self.locked_times),
      command_view=self.command_view,
      crossing_method="Incoming measured segments <=50ms; arrival clock; no extrapolation",
      phase_events=self.events,
      source_world=self.source_world,
      display_world="hope_table",
      simulation_world_translation=self.offset.tolist(),
      ball_clock="local packet arrival_time; causal sample-and-hold",
      replay="measured q/root pose; no policy or physics integration",
    )


class Timeline:
  def __init__(self, recording, start=0.0, rate=1.0, paused=False):
    self.recording = recording
    self.position = float(np.clip(start, 0, recording.duration))
    self.speed = PlaybackControl(rate)
    self.paused = paused
    self.lock = threading.RLock()

  def advance(self, elapsed):
    with self.lock:
      if not self.paused:
        self.position = min(
          self.recording.duration, self.position + elapsed * self.speed.rate
        )
        if self.position == self.recording.duration:
          self.paused = True
      return self.position

  def step_frames(self, count):
    """Seek by measured robot samples, including irregular recording intervals."""
    with self.lock:
      index = self.recording.index(self.position)
      index = int(np.clip(index + count, 0, len(self.recording.times) - 1))
      self.position = self.recording.times[index]
      self.paused = True

  def tts_status(self):
    """Causal timing on the recording clock; no wall-clock countdown or oracle."""
    with self.lock:
      if not hasattr(self.recording, "command"):
        return None
      command = self.recording.command(self.position)
      if command is None:
        return None
      start, deadline = command.get("received"), command.get("deadline")
      if not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in (start, deadline)
      ):
        return None
      result = dict(
        start=start,
        deadline=deadline,
        initial=deadline - start,
        elapsed=self.position - start,
        remaining=deadline - self.position,
      )
      encounter = self.recording.encounter(self.position, command)
      if encounter is not None:
        result.update(
          event_kind=encounter["kind"],
          event_time=encounter["time"],
          event_error=encounter["time"] - deadline,
        )
      return result

  def key(self, key):
    with self.lock:
      if self.speed.key(key):
        return
      if key == 32:
        self.paused = not self.paused
      elif key in (ord("R"), ord("r")):
        self.position = 0.0
        self.paused = False
      elif key in (263, 262):  # left/right: seek one second
        self.position += -1 if key == 263 else 1
      elif key in (268, 269):  # home/end
        self.position = 0 if key == 268 else self.recording.duration
      elif key in (ord(","), ord(".")):
        self.step_frames(-1 if key == ord(",") else 1)
      elif key in (ord("N"), ord("P")):
        events = self.recording.events
        i = (
          bisect.bisect_right(events, self.position + 1e-6)
          if key == ord("N")
          else bisect.bisect_left(events, self.position - 1e-6) - 1
        )
        self.position = events[int(np.clip(i, 0, len(events) - 1))]
      self.position = float(np.clip(self.position, 0, self.recording.duration))


class LogScene:
  def __init__(self, recording):
    import mujoco

    from deploy.hope_g1.scene import table_cfg
    from deploy.hope_g1.world import mujoco_ball_config
    from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
      get_g1_pingpong_robot_cfg,
    )
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    ball = recording.ball_config
    self.recording = recording
    self.model = Scene(
      SceneCfg(
        entities={
          "robot": get_g1_pingpong_robot_cfg(),
          "table": table_cfg(mujoco_ball_config(ball, recording.offset)),
        },
        terrain=TerrainEntityCfg(terrain_type="plane"),
      ),
      device="cpu",
    ).compile()
    self.data = mujoco.MjData(self.model)
    self.addresses = [
      self.model.jnt_qposadr[self.model.joint("robot/" + n).id] for n in recording.names
    ]
    free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
      raise ValueError("Expected one robot root free joint")
    self.root = self.model.jnt_qposadr[free[0]]
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [1.2, 0, 0.9]
    self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.5, 135, -22

  def update(self, seconds):
    import mujoco

    row = self.recording.rows[self.recording.index(seconds)]
    self.data.qpos[self.addresses] = row["q"]
    self.data.qpos[self.root : self.root + 3] = (
      np.asarray(row["root_pos"]) - self.recording.offset
    )
    quat = np.asarray(row["root_quat"])
    self.data.qpos[self.root + 3 : self.root + 7] = quat / np.linalg.norm(quat)
    self.data.time = seconds
    mujoco.mj_forward(self.model, self.data)
    return row

  def overlays(self, scene, seconds, row, *, clear=True):
    import mujoco

    if clear:
      scene.ngeom = 0

    def sphere(p, radius, color):
      if p is None or scene.ngeom >= scene.maxgeom:
        return
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(p) - self.recording.offset,
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
      )
      scene.ngeom += 1

    def arrow(p, direction, color, scale):
      v = vector(direction, 3)
      if v is None or scene.ngeom >= scene.maxgeom:
        return
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.ones(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
      )
      a = np.asarray(p) - self.recording.offset
      mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, a, a + v * scale)
      scene.ngeom += 1

    sphere(self.recording.ball(seconds), 0.02, [1, 0.65, 0.05, 1])
    # Trail includes only observations already delivered; never bridges missing data.
    start = bisect.bisect_left(self.recording.ball_times, seconds - 0.4)
    end = bisect.bisect_right(self.recording.ball_times, seconds)
    for i in range(start, end, 3):
      sphere(self.recording.ball_positions[i], 0.006, [1, 0.65, 0.05, 0.35])
    command = self.recording.command(seconds)
    # Preserve support for callers supplying a trace row directly.
    if command is None and not self.recording.command_times and received_command(row):
      command = dict(
        position=row.get("goal_position"),
        normal=row.get("goal_normal"),
        velocity=row.get("goal_velocity"),
        received=seconds,
      )
    goal = vector(command.get("position"), 3) if command else None
    if goal is not None:
      sphere(goal, 0.025, [0.2, 1, 0.3, 0.65])
      arrow(goal, command.get("normal"), [0.2, 1, 0.3, 1], 0.2)
      arrow(goal, command.get("velocity"), [0.2, 0.7, 1, 1], 0.06)
      # x-hit outline; it is the command plane, not a predicted ball path.
      if self.recording.command_times:
        ball = self.recording.ball_config
        center, width = ball["table_center_y"], ball["table_width"]
        corners = [
          np.array([goal[0], y, z])
          for y, z in [
            (center - width / 2, 0),
            (center + width / 2, 0),
            (center + width / 2, 1.2),
            (center - width / 2, 1.2),
          ]
        ]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
          arrow(a, b - a, [0.5, 0.7, 1, 0.4], 1)
      encounter = self.recording.encounter(seconds, command)
      if encounter is not None:
        actual = encounter["position"]
        sphere(actual, 0.035, [1, 0.15, 0.8, 1])
        arrow(goal, actual - goal, [1, 0.15, 0.8, 1], 1)
        racket = encounter.get("racket_position")
        if racket is not None:
          sphere(racket, 0.025, [0, 1, 1, 1])
          arrow(racket, actual - racket, [0, 1, 1, 1], 1)
          arrow(goal, racket - goal, [1, 0.7, 0.1, 1], 1)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--recording",
    "--log-dir",
    required=True,
    type=Path,
    help="Folder containing summary.json, trace.jsonl and optional mocap_packets.jsonl",
  )
  parser.add_argument("--rate", type=float, default=1.0)
  parser.add_argument(
    "--command-view",
    choices=("locked", "received"),
    default="locked",
    help="locked: actual selected goal; received: continuously updated UDP candidates",
  )
  parser.add_argument(
    "--start", type=float, default=0.0, help="Seconds from first usable robot frame"
  )
  parser.add_argument("--paused", action="store_true")
  parser.add_argument(
    "--rally-config",
    type=Path,
    help="Fallback court geometry/translation for legacy logs without these metadata",
  )
  parser.add_argument(
    "--assume-hope-table-coordinates",
    action="store_true",
    help=(
      "Explicitly treat a world-labeled log as HOPE tabletop coordinates "
      "without changing XYZ; only use when its source transform is identity"
    ),
  )
  parser.add_argument(
    "--inspect", action="store_true", help="Inspect recording without opening a window"
  )
  parser.add_argument(
    "--video", type=Path, help="Export an MP4 instead of opening a window"
  )
  parser.add_argument("--end", type=float, help="Video end time in recording seconds")
  parser.add_argument("--fps", type=int, default=30)
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(parser)
  args = parser.parse_args()
  configure_viewer(args)
  if not math.isfinite(args.start) or args.start < 0 or args.fps <= 0:
    parser.error("start must be finite/nonnegative and fps positive")
  recording = Recording(
    args.recording,
    rally_config=args.rally_config,
    assume_hope_table_coordinates=args.assume_hope_table_coordinates,
    command_view=args.command_view,
  )
  timeline = Timeline(recording, args.start, args.rate, args.paused)
  print(json.dumps(recording.describe(), indent=2))
  if args.inspect:
    return
  import mujoco

  scene = LogScene(recording)
  if args.video:
    import imageio.v2 as imageio

    end = recording.duration if args.end is None else min(args.end, recording.duration)
    if not math.isfinite(end) or end <= args.start:
      parser.error("Video end must be greater than start")
    args.video.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(scene.model, height=720, width=1280) as renderer:
      with imageio.get_writer(str(args.video), fps=args.fps) as writer:
        for t in np.arange(args.start, end, args.rate / args.fps):
          row = scene.update(t)
          renderer.update_scene(scene.data, camera=scene.camera)
          scene.overlays(renderer.scene, t, row, clear=False)
          writer.append_data(renderer.render())
    print(f"Saved {args.video}")
    return
  from deploy import viewer as deployment_viewer

  print(
    f"{HELP}; R restart; SPACE pause/resume; arrows seek 1s; ,/. step; N/P phase; HOME/END"
  )
  with deployment_viewer.launch_passive(
    scene.model, scene.data, key_callback=timeline.key
  ) as viewer:
    viewer.cam.lookat[:] = scene.camera.lookat
    viewer.cam.distance = scene.camera.distance
    viewer.cam.azimuth = scene.camera.azimuth
    viewer.cam.elevation = scene.camera.elevation
    previous = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      t = timeline.advance(now - previous)
      previous = now
      with viewer.lock():
        row = scene.update(t)
        scene.overlays(viewer.user_scn, t, row)
      ball = recording.ball(t)
      ball_text = (
        "missing/stale" if ball is None else np.array2string(ball, precision=3)
      )
      command = recording.command(t)
      command_tts = "--"
      command_text = f"no command ({recording.command_view} view)"
      crossing_text = "waiting for contact evidence / measured crossing"
      racket_text = "--"
      errors_text = "--"
      if command is not None:
        goal = np.asarray(command["position"])
        command_tts = (
          f"{command['deadline'] - t:.3f}"
          if "deadline" in command
          else row.get("time_to_strike")
        )
        command_text = f"{command['source']}: {np.array2string(goal, precision=3)}"
        encounter = recording.encounter(t, command)
        if encounter is not None:
          point = encounter["position"]
          error = np.linalg.norm(point - goal)
          label = (
            "ESTIMATED HIT (ball center)"
            if encounter["kind"] == "estimated_hit"
            else "MEASURED CROSSING (contact unconfirmed)"
          )
          crossing_text = (
            f"{label}: {np.array2string(point, precision=3)} @ {encounter['time']:.3f}s"
          )
          racket = encounter.get("racket_position")
          if racket is not None:
            racket_text = np.array2string(racket, precision=3)
            errors_text = (
              f"ball-command {error * 100:.1f}cm; "
              f"racket-ball {np.linalg.norm(racket - point) * 100:.1f}cm; "
              f"racket-command {np.linalg.norm(racket - goal) * 100:.1f}cm"
            )
          else:
            errors_text = (
              f"ball-command {error * 100:.1f}cm; racket measurement missing"
            )
      viewer.set_texts(
        (
          None,
          None,
          "RECORDED REAL STATE\nTime / speed\nState / phase / motion\nBall (table world)\nCommand TTS (s)\nDisplayed command XYZ\nHit / crossing XYZ\nRacket XYZ at same event time\nEvent errors (center distances)\nRobot frame age\nLegend\nControls",
          f"{recording.directory.name}\n{t:.2f}/{recording.duration:.2f}s  {timeline.speed.rate:g}x {'PAUSED' if timeline.paused else ''}\n"
          f"{row['state']} / {row['phase']} / {row['motion_index']}\n{ball_text}\n{command_tts}\n{command_text}\n{crossing_text}\n{racket_text}\n{errors_text}\n{t - recording.times[recording.index(t)]:.3f}s\n"
          "orange: live ball; green: command; blue: velocity; magenta: hit/crossing; cyan: racket at event\nR restart; SPACE pause/resume; [ ] speed; arrows seek 1s; , . step; N/P phase",
        )
      )
      viewer.sync()
      time.sleep(1 / 120)


if __name__ == "__main__":
  main()

"""Read-only ROS planner flight recorder; no command publishers or robot IO.

Run with the system Python after sourcing the HOPE ROS workspace. All readers
use BEST_EFFORT to avoid introducing reliable-reader backpressure. Observations
locate a failed boundary; they cannot prove an internal Python/DDS deadlock.
"""

import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo


def clean(value):
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {k: clean(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [clean(v) for v in value]
  return value


def encode(value):
  return json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def local_time():
  return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def writer(directory, name, size_mb, backups):
  logger = logging.Logger(name)
  handler = RotatingFileHandler(directory / name, maxBytes=size_mb * 1024**2,
                                backupCount=backups, encoding="utf-8")
  handler.setFormatter(logging.Formatter("%(message)s"))
  logger.addHandler(handler)
  return logger


def classify(*, uptime, planner_present, pose_age, raw_age, diag_age,
             progress_age, command_age, valid_progress_age, inbound_points,
             monitor_gap):
  """Conservative boundary diagnosis; no command during idle is not a stall."""
  if monitor_gap > 3:
    return "monitor_delayed", "warning", "监测器自身调度延迟，当前窗口不能据此判定 planner 卡住"
  if uptime < 10:
    return "warming_up", "info", "等待节点发现和首批样本"
  if not planner_present and diag_age > 2:
    return "waiting_for_planner", "info", "未发现本机 planner 进程或新诊断；继续等待启动"
  if pose_age > 1:
    if raw_age < 1:
      return "relay_input_gap", "warning", "原始球话题仍更新，但 /poses 中断：检查 relay 或该话题传输"
    return "ball_input_gap", "warning", "球输入中断：可能是球丢失、动捕停止或输入链路异常"
  if diag_age > 2:
    return "planner_heartbeat_gap", "warning", "球输入正常，planner 诊断停止：疑似执行器阻塞或诊断链路异常"
  if progress_age > 3:
    return "planner_receive_stalled", "warning", "诊断正常但 poses_received 不增长：检查 planner 订阅链路"
  if command_age > 2 and valid_progress_age < 2:
    return "command_delivery_gap", "warning", "planner 有效命令计数增长但未收到输出：检查发布及传输链路"
  if command_age > 10 and inbound_points >= 20:
    return "inbound_without_commands", "warning", "观察到来球但长期无命令；须结合门控原因确认，并非已证实故障"
  if command_age < 2:
    return "commands_observed", "info", "正在收到命令；有效项数量见记录"
  return "observing_no_commands", "info", "持续观测中；没有充分证据判定停发，可能正在等待合适来球"


def processes():
  """Inspect only local planner/relay processes; never signal or attach to them."""
  result = []
  for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
      continue
    try:
      argv = (entry / "cmdline").read_bytes().split(b"\0")
      names = {Path(a.decode(errors="replace")).name for a in argv[:3]}
      if "hope_planner_command_array_node" in names:
        role = "planner"
      elif "pose_relay" in names:
        role = "relay"
      elif "client_node" in names and b"vrpn_mocap" in b" ".join(argv[:3]):
        role = "vrpn"
      else:
        continue
      fields = {}
      for line in (entry / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("State", "VmRSS", "Threads"):
          fields[key] = value.strip()
      stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
      result.append(dict(pid=int(entry.name), role=role, status=fields,
                         cpu_ticks=int(stat[11]) + int(stat[12]),
                         start_ticks=int(stat[19]),
                         wchan=(entry / "wchan").read_text().strip()))
    except (OSError, ValueError, IndexError):
      continue
  return result


def thread_snapshot(procs):
  result = []
  for proc in procs:
    if proc["role"] != "planner":
      continue
    try:
      for task in (Path("/proc") / str(proc["pid"]) / "task").iterdir():
        try:
          result.append(dict(pid=proc["pid"], tid=int(task.name),
                             comm=(task / "comm").read_text().strip(),
                             wchan=(task / "wchan").read_text().strip()))
        except OSError:
          pass
    except OSError:
      pass
  return result


class Recorder:
  def __init__(self, directory, config, node):
    self.directory, self.config, self.node = directory, config, node
    self.trace = writer(directory, "trace.jsonl", 32, 7)
    self.summary = writer(directory, "summary.jsonl", 8, 3)
    self.incidents = writer(directory, "incidents.jsonl", 16, 7)
    self.events = writer(directory, "events.jsonl", 4, 3)
    self.start = self.tick_time = time.monotonic()
    self.counts, self.previous_counts = Counter(), Counter()
    self.last = {}
    self.ring = deque(maxlen=16000)
    self.inbound = deque(maxlen=4000)
    self.reasons = deque(maxlen=1000)
    self.prev_pose = None
    self.diag = {}
    self.diag_received = self.diag_valid = None
    self.progress_time = self.start
    self.valid_progress_time = -math.inf
    self.prev_source_stamp = {}
    self.timestamp_anomalies = Counter()
    self.latencies = deque(maxlen=2000)
    self.procs, self.graph = [], {}
    self.planner_identity = ()
    self.generation_start = self.start
    self.mode = None
    self.candidate_mode, self.candidate_since = None, self.start
    self.last_incident = -math.inf
    self.incident_id = None
    self.capture_until = -math.inf
    self.last_graph = -math.inf
    self.last_console = -math.inf
    self.config_hash = None

  def record(self, kind, data):
    now = time.monotonic()
    event = dict(kind=kind, monotonic=now, wall_time=time.time(), **data)
    self.counts[kind] += 1
    self.last[kind] = now
    self.trace.info(encode(event))
    self.ring.append(event)
    while self.ring and now - self.ring[0]["monotonic"] > 15:
      self.ring.popleft()
    if now < self.capture_until:
      self.incidents.info(encode(dict(incident_id=self.incident_id, phase="after", sample=event)))
    return now

  def position(self, kind, stamp, xyz):
    previous = self.prev_source_stamp.get(kind)
    if previous is not None and stamp <= previous:
      self.timestamp_anomalies[kind + ("_duplicate" if stamp == previous else "_backward")] += 1
    self.prev_source_stamp[kind] = stamp
    now = self.record(kind, dict(source_stamp=stamp, position=xyz,
                                 source_age_s=time.time() - stamp))
    if kind == "poses":
      if self.prev_pose is not None:
        old_t, old_p = self.prev_pose
        dt = stamp - old_t
        if (0 < dt < .1 and all(math.isfinite(v) for v in xyz + old_p)
            and 0 <= xyz[0] <= 2.2 and -1.5 <= xyz[1] <= 0 and xyz[2] > .02
            and -.1 > (xyz[0] - old_p[0]) / dt > -30):
          self.inbound.append(now)
      self.prev_pose = (stamp, xyz)

  def diagnostics(self, msg):
    for status in msg.status:
      if status.name != "hope_planner_command_array":
        continue
      values = {v.key: v.value for v in status.values}
      # ROS Humble maps diagnostic_msgs' byte level to a one-byte bytes value.
      level = int.from_bytes(status.level, "little") if isinstance(status.level, bytes) else int(status.level)
      now = self.record("diagnostics", dict(level=level, message=status.message, values=values))
      self.diag = values
      self.reasons.append((now, values.get("command_reason", "unknown")))
      try:
        received, valid = int(values["poses_received"]), int(values["valid_commands"])
      except (ValueError, KeyError):
        continue
      if self.diag_received is not None and received < self.diag_received:
        self.generation_start = now
        self.valid_progress_time = -math.inf
      if received != self.diag_received:
        self.progress_time = now
      # The first cumulative count is a baseline, not a new publication.
      if self.diag_valid is not None and valid > self.diag_valid:
        self.valid_progress_time = now
      self.diag_received, self.diag_valid = received, valid

  def commands(self, msg):
    def stamp(s):
      return s.sec + s.nanosec * 1e-9
    now_wall = time.time()
    output_stamp = stamp(msg.header.stamp)
    commands = [dict(valid=c.valid, source_ball_time=c.source_ball_time,
                     position=[c.position.x, c.position.y, c.position.z],
                     source_ball_position=[c.source_ball_position.x, c.source_ball_position.y, c.source_ball_position.z],
                     time_to_strike=c.time_to_strike, strike_time=c.strike_time,
                     stage1_fit_rmse_m=c.stage1_fit_rmse_m) for c in msg.commands]
    source_stamp = commands[0]["source_ball_time"] if commands else None
    now = self.record("commands", dict(publish_stamp=output_stamp, source_stamp=source_stamp,
                                       center_index=msg.center_index, commands=commands))
    if any(c["valid"] for c in commands):
      self.counts["valid_groups"] += 1
      self.last["valid_groups"] = now
    if source_stamp is not None:
      self.latencies.append((now, output_stamp - source_stamp, now_wall - output_stamp))

  def capture_metadata(self):
    try:
      raw = self.config.read_bytes()
      digest = hashlib.sha256(raw).hexdigest()
      if digest != self.config_hash:
        self.config_hash = digest
        self.events.info(encode(dict(kind="config_file_snapshot", time=local_time(),
                                     path=str(self.config), sha256=digest,
                                     text=raw.decode(errors="replace"),
                                     note="磁盘配置快照；不代表运行进程已经加载此版本")))
    except OSError:
      pass

  def tick(self):
    now = time.monotonic()
    elapsed = now - self.tick_time
    self.tick_time = now
    if now - self.last_graph >= 5:
      self.procs = processes()
      identity = tuple(sorted((p["pid"], p["start_ticks"]) for p in self.procs if p["role"] == "planner"))
      if identity != self.planner_identity:
        self.events.info(encode(dict(kind="planner_process_change", time=local_time(),
                                     previous=self.planner_identity, current=identity)))
        self.planner_identity = identity
        self.generation_start = now
        self.diag_received = self.diag_valid = None
        self.progress_time = now
        self.valid_progress_time = -math.inf
      self.graph = {topic: self.node.count_publishers(topic) for topic in
                    ("/poses", "/racket/commands", "/planner/diagnostics", "/vrpn_mocap/MCAvatar/pose30310")}
      self.capture_metadata()
      self.last_graph = now
    age = lambda kind: now - self.last.get(kind, -math.inf)
    while self.inbound and now - self.inbound[0] > 10:
      self.inbound.popleft()
    reasons = Counter(reason for t, reason in self.reasons if now - t <= 10)
    state, level, explanation = classify(
      uptime=now - self.generation_start, planner_present=bool(self.planner_identity),
      pose_age=age("poses"), raw_age=age("raw_ball"), diag_age=age("diagnostics"),
      progress_age=now - self.progress_time, command_age=age("commands"),
      valid_progress_age=now - self.valid_progress_time,
      inbound_points=len(self.inbound), monitor_gap=elapsed)
    if sum(p["role"] == "planner" for p in self.procs) > 1 or self.graph.get("/poses", 0) > 1:
      state, level, explanation = "multiple_sources", "warning", "发现多个 planner 进程或 /poses 发布者，须检查混流"
    rates = {k: round((n - self.previous_counts[k]) / max(elapsed, 1e-6), 2)
             for k, n in self.counts.items()}
    self.previous_counts = self.counts.copy()
    latencies = [(source, transport) for t, source, transport in self.latencies if now - t <= 10]
    snapshot = dict(time=local_time(), monotonic=now, monitor_pid=os.getpid(),
                    state=state, level=level, explanation=explanation,
                    monitor_interval_s=elapsed, rates_hz=rates, totals=dict(self.counts),
                    ages_s={k: age(k) for k in ("raw_ball", "poses", "ball_point", "diagnostics", "commands", "valid_groups")},
                    planner_receive_progress_age_s=now - self.progress_time,
                    inbound_points_last_10s=len(self.inbound), diagnostics=self.diag,
                    diagnostic_reason_samples_last_10s=dict(reasons),
                    timestamp_anomalies=dict(self.timestamp_anomalies),
                    latency_last_10s=dict(samples=len(latencies),
                      max_source_to_publish_s=max((v[0] for v in latencies), default=None),
                      max_publish_to_monitor_s=max((v[1] for v in latencies), default=None)),
                    processes=self.procs, publisher_counts=self.graph,
                    limits="话题观察只能定位异常边界；wchan 不是 Python 栈，不能直接证明 DDS/计算死锁。时间差依赖源时钟一致。")
    self.summary.info(encode(snapshot))
    temp = self.directory / "status.tmp"
    temp.write_text(json.dumps(clean(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(self.directory / "status.json")
    if state != self.candidate_mode:
      self.candidate_mode, self.candidate_since = state, now
    # Require a persistent warning; preserve intermittent motion as evidence,
    # without treating every short quiet period as a robot fault.
    if state != self.mode and (level == "info" or now - self.candidate_since >= 2):
      self.events.info(encode(dict(kind="state_change", **snapshot)))
      self.mode = state
      print(f"{snapshot['time']} {level.upper()} {state}: {explanation}", flush=True)
    if level == "warning" and now - self.candidate_since >= 2 and now - self.last_incident >= 60:
      self.incident_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
      self.last_incident = now
      self.incidents.info(encode(dict(incident_id=self.incident_id, kind="trigger",
                                      snapshot=snapshot, threads=thread_snapshot(self.procs))))
      for event in self.ring:
        self.incidents.info(encode(dict(incident_id=self.incident_id, phase="before", sample=event)))
      self.capture_until = now + 15
    if now - self.last_console >= 60:
      print(f"{snapshot['time']} heartbeat state={state} rates={rates}", flush=True)
      self.last_console = now


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=Path("/home/lxz/HOPE/hope_ws/src/hope_planner_command_array/config/hope_planner.yaml"))
  parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until stopped")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  # Prevent accidental duplicate monitors writing the same rotating files.
  import fcntl
  lock = (args.output / "monitor.lock").open("w")
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  import rclpy
  from rclpy.node import Node
  from rclpy.qos import QoSProfile, ReliabilityPolicy
  from geometry_msgs.msg import PoseArray, PoseStamped, PointStamped
  from diagnostic_msgs.msg import DiagnosticArray
  from hope_msgs.msg import RacketCommandArray

  rclpy.init(args=[])
  node = Node("planner_readonly_monitor")
  recorder = Recorder(args.output, args.config, node)
  qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
  def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
  def position(kind, msg, p):
    recorder.position(kind, stamp(msg), [p.x, p.y, p.z])
  def poses(msg):
    if msg.poses:
      position("poses", msg, msg.poses[0].position)
    else:
      recorder.record("empty_poses", dict(source_stamp=stamp(msg)))
  subscriptions = [
    node.create_subscription(PoseArray, "/poses", poses, qos),
    node.create_subscription(PoseStamped, "/vrpn_mocap/MCAvatar/pose30310",
                             lambda m: position("raw_ball", m, m.pose.position), qos),
    node.create_subscription(PointStamped, "/ball/point",
                             lambda m: position("ball_point", m, m.point), qos),
    node.create_subscription(DiagnosticArray, "/planner/diagnostics", recorder.diagnostics, qos),
    node.create_subscription(RacketCommandArray, "/racket/commands", recorder.commands, qos),
  ]
  node.create_timer(1, recorder.tick)
  (args.output / "monitor.pid").write_text(str(os.getpid()) + "\n")
  recorder.events.info(encode(dict(kind="monitor_start", time=local_time(), pid=os.getpid(),
    environment={k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION")},
    log_limit_mb=432, note="日志轮转：trace 256 MiB，summary 32 MiB，incidents 128 MiB，events 16 MiB")))
  deadline = time.monotonic() + args.duration if args.duration > 0 else math.inf
  try:
    while rclpy.ok() and time.monotonic() < deadline:
      rclpy.spin_once(node, timeout_sec=.25)
  except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
    pass
  finally:
    recorder.events.info(encode(dict(kind="monitor_stop", time=local_time())))
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()
    logging.shutdown()
    lock.close()


if __name__ == "__main__":
  main()

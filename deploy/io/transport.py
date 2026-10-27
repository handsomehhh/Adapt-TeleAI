"""Local UDP handoff so ROS and policy inference may use separate Python envs."""

import argparse
import csv
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

from deploy.g1_pingpong.perception import (
  BallEstimator,
  RootPoseCache,
  sample_received_time,
)
from deploy.io.diagnostics import BufferedWriter, json_default


class UDPMocap:
  def __init__(self, config):
    self.cfg = config
    self.recorder = None
    self.ball = BallEstimator(**config["estimator"])
    self.root = RootPoseCache(
      config["root_marker_to_pelvis"],
      config["root_stale_timeout"],
      config.get("root_marker_to_pelvis_quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)),
      conditioning=config.get("root_pose_filter"),
    )
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.socket.bind((config["udp"]["host"], config["udp"]["port"]))
    self.socket.settimeout(0.1)
    if config.get("diagnostic_output_dir"):
      self.recorder = BufferedWriter(
        Path(config["diagnostic_output_dir"]) / "mocap_packets.jsonl"
      )
    self.stop = threading.Event()
    self.rejected = 0
    self.accepted = 0
    self._planner_lock = threading.Lock()
    self._planner_packet = None
    self._planner_received = None
    self.last_error = None
    self.thread = threading.Thread(target=self._receive, daemon=True)
    self.thread.start()

  def accept(self, message, *, wall_time=None, monotonic_time=None):
    wall_time = time.time() if wall_time is None else wall_time
    monotonic_time = time.monotonic() if monotonic_time is None else monotonic_time
    recorder = getattr(self, "recorder", None)
    if recorder is not None:
      recorder.write(
        json.dumps(
          {"arrival_time": monotonic_time, "wall_time": wall_time, "packet": message},
          default=json_default,
        )
        + "\n"
      )
    if message["frame"] != self.cfg["world_frame"]:
      raise ValueError("Mocap frame differs from deployment world")
    stamp = float(message["stamp"])
    if message["kind"] == "ball":
      received = sample_received_time(
        stamp, self.cfg, wall_time=wall_time, monotonic_time=monotonic_time
      )
      self.ball.update(message["position"], stamp, received=received)
    elif message["kind"] == "root":
      received = sample_received_time(
        stamp, self.cfg, wall_time=wall_time, monotonic_time=monotonic_time
      )
      self.root.update(
        message["position"], message["quaternion_wxyz"], stamp, received=received
      )
    elif message["kind"] == "racket_command":
      vectors = [
        np.asarray(message[key], dtype=float)
        for key in ("position", "velocity", "normal")
      ]
      tts = float(message["time_to_strike"])
      if (
        any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors)
        or not np.isfinite(tts)
      ):
        raise ValueError("Invalid planner racket command")
      with self._planner_lock:
        self._planner_packet = dict(message)
        self._planner_received = monotonic_time
    else:
      raise ValueError("Unknown mocap message kind")
    self.accepted += 1

  def latest_planner_command(self):
    """Return the most recently received ROS command and its local arrival time."""
    with self._planner_lock:
      if self._planner_packet is None:
        return None, None
      return dict(self._planner_packet), self._planner_received

  def _receive(self):
    while not self.stop.is_set():
      try:
        payload, _ = self.socket.recvfrom(4096)
        self.accept(json.loads(payload))
      except TimeoutError:
        continue
      except (ValueError, KeyError, TypeError) as exc:
        self.rejected += 1
        self.last_error = str(exc)
      except OSError:
        if not self.stop.is_set():
          self.rejected += 1
        break

  def diagnostics(self):
    now = time.monotonic()
    ball, root = self.ball.get(now), self.root.get(now)
    planner_packet, planner_received = self.latest_planner_command()
    result = {
      "accepted": self.accepted,
      "rejected": self.rejected,
      "last_error": self.last_error,
      "ball_valid": ball is not None,
      "root_valid": root is not None,
      "ball_position": ball.position.tolist() if ball else None,
      "ball_velocity": ball.velocity.tolist() if ball else None,
      "root_position": root[0].tolist() if root else None,
      "planner_command_valid": bool(planner_packet and planner_packet.get("valid")),
      "planner_command_age": (
        now - planner_received if planner_received is not None else None
      ),
      "planner_command_time_to_strike": (
        planner_packet.get("time_to_strike") if planner_packet else None
      ),
      "planner_command_position": (
        planner_packet.get("position") if planner_packet else None
      ),
    }
    recorder = getattr(self, "recorder", None)
    if recorder is not None:
      result["recording"] = recorder.diagnostics()
    return result

  def close(self):
    self.stop.set()
    self.socket.close()
    self.thread.join(timeout=2)
    if self.recorder is not None:
      self.recorder.close()


def build_mocap(config):
  if config["transport"] == "udp":
    return UDPMocap(config)
  if config["transport"] == "ros2":
    from deploy.io.mocap import RosMocap

    return RosMocap(config)
  raise ValueError("mocap transport must be udp or ros2")


def replay_udp(config, csv_path):
  """Replay normalized ball/root recordings without ROS or robot dependencies."""
  with Path(csv_path).open() as stream:
    rows = list(csv.DictReader(stream))
  samples = np.array(
    [[float(row[key]) for key in ("t", "x", "y", "z")] for row in rows]
  )
  if (
    len(samples) < 3
    or not np.isfinite(samples).all()
    or np.any(np.diff(samples[:, 0]) <= 0)
  ):
    raise ValueError("Expected finite, increasing t,x,y,z CSV")
  root_keys = ["root_" + axis for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
  has_root = all(key in rows[0] for key in root_keys)
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  start, epoch = time.monotonic(), time.time() - config["source_clock_offset"]
  try:
    for row, (t, x, y, z) in zip(rows, samples, strict=True):
      offset = t - samples[0, 0]
      time.sleep(max(start + offset - time.monotonic(), 0))
      packets = [
        {
          "kind": "ball",
          "frame": config["world_frame"],
          "stamp": epoch + offset,
          "position": [x, y, z],
        }
      ]
      if has_root:
        values = [float(row[key]) for key in root_keys]
        packets.append(
          {
            "kind": "root",
            "frame": config["world_frame"],
            "stamp": epoch + offset,
            "position": values[:3],
            "quaternion_wxyz": values[3:],
          }
        )
      for packet in packets:
        sock.sendto(
          json.dumps(packet, allow_nan=False).encode(),
          (config["udp"]["host"], config["udp"]["port"]),
        )
  finally:
    sock.close()


def main():
  from deploy.io.mocap import read_config

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=["monitor", "replay"])
  parser.add_argument("--config", type=Path, default=Path("deploy/configs/mocap.yaml"))
  parser.add_argument("--csv", type=Path)
  parser.add_argument("--duration", type=float, default=10.0)
  args = parser.parse_args()
  config, _ = read_config(args.config)
  if args.mode == "replay":
    if args.csv is None:
      parser.error("replay requires --csv")
    replay_udp(config, args.csv)
    return
  if args.duration <= 0:
    parser.error("duration must be positive")
  receiver = UDPMocap(config)
  start = time.monotonic()
  try:
    while time.monotonic() - start < args.duration:
      print(json.dumps(receiver.diagnostics(), allow_nan=False))
      time.sleep(min(1.0, args.duration))
  finally:
    receiver.close()


if __name__ == "__main__":
  main()

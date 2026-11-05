"""Typed G1 planner ROS boundary; pure packet conversion remains testable offline."""

import json
import socket
import time

import numpy as np


def command_packet(msg, cfg, *, ros_now, monotonic_now):
  if msg.header.frame_id != cfg["planner_command_source_frame"]:
    raise ValueError("G1 planner command frame mismatch")

  def xyz(v):
    return [float(v.x), float(v.y), float(v.z)]

  remaining = float(msg.strike_time) - ros_now
  packet = dict(
    kind="racket_command",
    schema="g1_1v1_v1",
    frame=cfg["world_frame"],
    stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
    bridge_monotonic=monotonic_now,
    time_to_strike=remaining,
    runtime_mode=msg.runtime_mode,
    physics_sha256=msg.physics_sha256,
    contract_sha256=msg.contract_sha256,
    session_id=msg.session_id,
    command_id=int(msg.command_id),
    state_sequence=int(msg.state_sequence),
    valid=bool(msg.valid),
    reason=msg.reason,
    side=int(msg.side),
    position=xyz(msg.position),
    velocity=xyz(msg.velocity),
    normal=xyz(msg.normal),
    station=xyz(msg.station)[:2],
    outgoing_velocity=xyz(msg.ball_velocity_outgoing),
    outgoing_velocity_requested=xyz(msg.ball_velocity_requested),
    outgoing_velocity_residual=float(msg.outgoing_velocity_residual),
    source_ball_time=float(msg.source_ball_time),
    source_ball_position=xyz(msg.source_ball_position),
    predicted_bounces=int(msg.total_bounces),
  )
  numbers = np.r_[
    packet["position"],
    packet["velocity"],
    packet["normal"],
    packet["station"],
    packet["outgoing_velocity"],
    packet["outgoing_velocity_requested"],
    packet["source_ball_position"],
    packet["source_ball_time"],
    remaining,
    packet["stamp"],
    monotonic_now,
    packet["outgoing_velocity_residual"],
  ]
  if not np.isfinite(numbers).all():
    raise ValueError("Nonfinite G1 planner message")
  return packet


def subscribe(node, cfg, consume):
  from hope_msgs.msg import G1RacketCommand

  def receive(msg):
    try:
      consume(
        command_packet(
          msg,
          cfg,
          ros_now=node.get_clock().now().nanoseconds * 1e-9,
          monotonic_now=time.monotonic(),
        )
      )
    except (ValueError, TypeError, KeyError) as exc:
      node.get_logger().error(str(exc))

  return node.create_subscription(
    G1RacketCommand, cfg["planner_command_topic"], receive, 1
  )


def feedback_relay(node, cfg):
  """Receive local controller target state, publish it in the ROS clock domain."""
  from hope_msgs.msg import G1PlannerState

  options = cfg["planner_feedback"]
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind((options["host"], options["port"]))
  sock.setblocking(False)
  publisher = node.create_publisher(G1PlannerState, options["topic"], 1)

  def pump():
    latest = None
    # Bounded drain: controller feedback is at 50 Hz; retain the newest state.
    for _ in range(64):
      try:
        payload, _ = sock.recvfrom(4096)
        latest = json.loads(payload)
      except BlockingIOError:
        break
      except (ValueError, UnicodeError):
        continue
    if latest is None:
      return
    try:
      age = time.monotonic() - float(latest["generated_monotonic"])
      if not 0 <= age <= 0.15:
        return
      station, home = np.asarray(latest["station"]), np.asarray(latest["home"])
      if (
        station.shape != (2,)
        or home.shape != (2,)
        or not np.isfinite(np.r_[station, home]).all()
      ):
        raise ValueError("Invalid planner-state station")
      msg = G1PlannerState()
      msg.header.frame_id = cfg["planner_command_source_frame"]
      ns = node.get_clock().now().nanoseconds - round(age * 1e9)
      msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
      msg.contract_sha256, msg.session_id = (
        latest["contract_sha256"],
        latest["session_id"],
      )
      msg.runtime_mode = latest["runtime_mode"]
      msg.physics_sha256 = latest["physics_sha256"]
      msg.sequence, msg.side = int(latest["sequence"]), int(latest["side"])
      msg.enabled = bool(latest["enabled"])
      msg.station.x, msg.station.y = map(float, station)
      msg.home.x, msg.home.y = map(float, home)
      publisher.publish(msg)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
      node.get_logger().error(f"G1 feedback rejected: {exc}")

  node.create_timer(0.005, pump)
  return sock

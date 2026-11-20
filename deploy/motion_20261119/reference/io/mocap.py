"""HOPE-compatible ROS 2 VRPN relay and asynchronous deployment subscriptions.

ROS imports are lazy so MuJoCo and offline checks need no ROS installation.
"""

import argparse
import csv
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from deploy.motion_20261119.reference.g1_pingpong.core import rotation
from deploy.motion_20261119.reference.planner_interface import COMMAND_ARRAY, SINGLE_COMMAND, PlannerInterfaceError, command_type
from deploy.motion_20261119.reference.g1_pingpong.perception import (
  BallEstimator,
  RootPoseCache,
  sample_received_time,
)


def stamp_seconds(stamp):
  return stamp.sec + stamp.nanosec * 1e-9


def read_config(path):
  cfg = yaml.safe_load(Path(path).read_text())
  r = rotation(cfg["transform"]["quaternion_wxyz"])
  scale = cfg["transform"]["scale"]
  translation = np.asarray(cfg["transform"]["translation"])
  if (
    not np.isfinite(scale)
    or scale <= 0
    or translation.shape != (3,)
    or not np.isfinite(translation).all()
  ):
    raise ValueError("Invalid mocap transform")
  return cfg, r


def input_packet(msg, kind, cfg):
  """Normalize exactly once: HOPE source frame -> G1 deployment world.

  Without `source`, inputs are already normalized by the legacy relay.
  Keep the acquisition timestamp; UDP transport must not refresh stale samples.
  """
  source = cfg.get("source")
  # Per-input frame labels do not imply a TF transform. Both inputs must still
  # share the coordinates expected by the configured common transform.
  expected = (
    source.get(f"{kind}_frame", source["frame"]) if source else cfg["world_frame"]
  )
  if msg.header.frame_id != expected:
    raise ValueError(f"Expected source frame {expected}, got {msg.header.frame_id}")
  message_type = source.get("ball_type", "point_stamped") if source else "point_stamped"
  pose = None
  if kind == "root" or message_type == "pose_stamped":
    pose = msg.pose
    p = pose.position
  elif message_type == "pose_array":
    index = source.get("ball_pose_index", 0)
    if not isinstance(index, int) or not 0 <= index < len(msg.poses):
      raise ValueError("Configured ball_pose_index is absent from /poses")
    p = msg.poses[index].position
  elif message_type == "point_stamped":
    p = msg.point
  else:
    raise ValueError(f"Unsupported ball message type: {message_type}")
  position = np.array([p.x, p.y, p.z], dtype=float)
  if not np.isfinite(position).all():
    raise ValueError("Non-finite mocap position")
  if source:
    transform_cfg = cfg["transform"]
    position = (
      rotation(transform_cfg["quaternion_wxyz"]) @ position * transform_cfg["scale"]
      + transform_cfg["translation"]
    )
  packet = {
    "kind": kind,
    "frame": cfg["world_frame"],
    "stamp": stamp_seconds(msg.header.stamp),
    "position": position.tolist(),
  }
  if kind == "root":
    assert pose is not None
    q = pose.orientation
    quaternion = np.array([q.w, q.x, q.y, q.z])
    rotation(quaternion)
    if source:
      quaternion = multiply(cfg["transform"]["quaternion_wxyz"], quaternion)
    packet["quaternion_wxyz"] = quaternion.tolist()
  return packet


def subscribe_positions(node, cfg, consume):
  """Subscribe only to the configured ball and robot topics, depth one."""
  from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
  from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

  source = cfg.get("source", {})
  types = {
    "point_stamped": PointStamped,
    "pose_stamped": PoseStamped,
    "pose_array": PoseArray,
  }
  ball_type = source.get("ball_type", "point_stamped")
  if ball_type not in types:
    raise ValueError(f"Unsupported ball message type: {ball_type}")
  qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
  )

  def receive(msg, kind):
    try:
      packet = input_packet(msg, kind, cfg)
      sample_received_time(packet["stamp"], cfg)
      consume(packet)
    except (ValueError, TypeError, KeyError) as exc:
      node.get_logger().error(str(exc))

  node.create_subscription(
    types[ball_type],
    source.get("ball_topic", cfg["ball_topic"]),
    lambda msg: receive(msg, "ball"),
    qos,
  )
  node.create_subscription(
    PoseStamped,
    source.get("root_topic", cfg["root_topic"]),
    lambda msg: receive(msg, "root"),
    qos,
  )


def subscribe_planner_commands(node, cfg, consume):
  """Forward ROS planner commands over the same local UDP handoff as mocap."""
  from hope_msgs.msg import RacketCommand
  from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

  qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
  )
  interface = command_type(cfg)
  message_class = RacketCommand
  if interface == COMMAND_ARRAY:
    from hope_msgs.msg import RacketCommandArray
    message_class = RacketCommandArray
  topic = cfg.get("planner_command_topic", "/racket/commands" if interface == COMMAND_ARRAY else "/racket/command")
  expected_frame = cfg.get("planner_command_source_frame", "world")

  def serialize(msg):
    if msg.header.frame_id != expected_frame:
      node.get_logger().error(
        f"Expected planner command frame {expected_frame}, got {msg.header.frame_id}"
      )
      return
    p, v, n, outgoing = (
      msg.position,
      msg.velocity,
      msg.normal,
      msg.ball_velocity_outgoing,
    )
    position = np.array([p.x, p.y, p.z], dtype=float)
    velocity = np.array([v.x, v.y, v.z], dtype=float)
    normal = np.array([n.x, n.y, n.z], dtype=float)
    outgoing_velocity = np.array([outgoing.x, outgoing.y, outgoing.z], dtype=float)
    tts = float(msg.time_to_strike)
    source_ball_time = float(msg.source_ball_time)
    source_ball_position = np.array(
      [msg.source_ball_position.x, msg.source_ball_position.y, msg.source_ball_position.z],
      dtype=float,
    )
    valid = bool(msg.valid)
    if (
      not np.isfinite(
        np.r_[position, velocity, normal, outgoing_velocity, source_ball_position, tts, source_ball_time]
      ).all()
    ):
      valid = False
      position = velocity = outgoing_velocity = np.zeros(3)
      normal = np.array([0.0, 0.0, 1.0])
      tts = 0.0
      source_ball_time = 0.0
      source_ball_position = np.zeros(3)
    packet = {
      "kind": "racket_command",
      "command_type": SINGLE_COMMAND,
      # The planner's world label is normalized to the deployment's already
      # calibrated HOPE table coordinates; XYZ is not transformed here.
      "frame": cfg["world_frame"],
      "stamp": stamp_seconds(msg.header.stamp),
      "bridge_monotonic": time.monotonic(),
      "source_ball_time": source_ball_time,
      "source_ball_position": source_ball_position.tolist(),
      "position": position.tolist(),
      "velocity": velocity.tolist(),
      "normal": normal.tolist(),
      "outgoing_velocity": outgoing_velocity.tolist(),
      "time_to_strike": tts,
      "valid": valid,
      "clears_net": bool(msg.clears_net),
      "bypasses_net_posts": bool(msg.bypasses_net_posts),
      "predicted_bounces": int(msg.predicted_bounces),
      "strike_time": float(msg.strike_time) if np.isfinite(msg.strike_time) else None,
      "stage1_fit_rmse_m": float(msg.stage1_fit_rmse_m) if np.isfinite(msg.stage1_fit_rmse_m) else None,
    }
    return packet

  def receive(msg):
    if interface == SINGLE_COMMAND:
      packet = serialize(msg)
    else:
      if msg.header.frame_id != expected_frame:
        raise ValueError(f"Expected planner command frame {expected_frame}, got {msg.header.frame_id}")
      published = time.monotonic()
      entries = [serialize(entry) for entry in msg.commands]
      if any(entry is None for entry in entries):
        return
      packet = dict(
        kind="racket_command_array", command_type=COMMAND_ARRAY,
        frame=cfg["world_frame"], stamp=stamp_seconds(msg.header.stamp),
        bridge_monotonic=published, commands=entries, center_index=int(msg.center_index),
        x_hit=float(msg.x_hit), forward_num_commands=int(msg.forward_num_commands),
        backward_num_commands=int(msg.backward_num_commands),
        forward_hit_interval=float(msg.forward_hit_interval),
        backward_hit_interval=float(msg.backward_hit_interval),
      )
    if packet is not None:
      consume(packet)

  def check_publishers():
    publishers = node.get_publishers_info_by_topic(topic)
    actual = {info.topic_type for info in publishers}
    # The two default endpoints are one logical planner input. Diagnose the
    # common case of starting the other interface instead of waiting forever.
    if not actual and topic in ("/racket/command", "/racket/commands"):
      other = "/racket/command" if topic == "/racket/commands" else "/racket/commands"
      actual = {info.topic_type for info in node.get_publishers_info_by_topic(other)}
    if actual and actual != {interface}:
      error = f"Planner interface mismatch: {topic} requires {interface}, active publisher types={sorted(actual)}"
      consume(dict(kind="planner_interface_error", frame=cfg["world_frame"], stamp=0.,
                   command_type=interface, error=error))
      raise PlannerInterfaceError(error)

  node.create_subscription(message_class, topic, receive, qos)
  node.create_timer(.5, check_publishers)


def multiply(q, p):
  w, v = q[0], np.asarray(q[1:])
  s, u = p[0], np.asarray(p[1:])
  return np.r_[w * s - v @ u, w * u + s * v + np.cross(v, u)]


class RosMocap:
  def __init__(self, config):
    import rclpy

    self.rclpy = rclpy
    self.owns_context = not rclpy.ok()
    if self.owns_context:
      rclpy.init()
    self.node = rclpy.create_node("g1_pingpong_mocap_input")
    self.ball = BallEstimator(**config["estimator"])
    self.root = RootPoseCache(
      config["root_marker_to_pelvis"],
      config["root_stale_timeout"],
      config.get("root_marker_to_pelvis_quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)),
      conditioning=config.get("root_pose_filter"),
    )

    def consume(packet):
      stamp = packet["stamp"]
      received = sample_received_time(stamp, config)
      if packet["kind"] == "ball":
        self.ball.update(packet["position"], stamp, received=received)
      else:
        self.root.update(
          packet["position"], packet["quaternion_wxyz"], stamp, received=received
        )

    subscribe_positions(self.node, config, consume)
    self.executor = rclpy.executors.SingleThreadedExecutor()
    self.executor.add_node(self.node)
    self.thread = threading.Thread(target=self.executor.spin, daemon=True)
    self.thread.start()

  def close(self):
    self.executor.shutdown(timeout_sec=2)
    self.thread.join(timeout=2)
    self.node.destroy_node()
    if self.owns_context:
      self.rclpy.shutdown()


def relay(config):
  """Named Ball/P1 VRPN poses -> calibrated /ball/point and /P1/pose.

  Requires an externally running vrpn_mocap client, as in HOPE. Explicit names
  avoid accidentally identifying another moving marker as the ball.
  """
  import rclpy
  from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
  from rclpy.qos import qos_profile_sensor_data

  cfg, r = read_config(config)
  rclpy.init()
  node = rclpy.create_node("g1_pingpong_vrpn_relay")
  ball_pub = node.create_publisher(
    PointStamped, cfg["ball_topic"], qos_profile_sensor_data
  )
  root_pub = node.create_publisher(
    PoseStamped, cfg["root_topic"], qos_profile_sensor_data
  )
  poses_pub = node.create_publisher(PoseArray, "/poses", qos_profile_sensor_data)
  root_pose = [None]

  def transform(msg):
    p, q = msg.pose.position, msg.pose.orientation
    position = (
      r @ np.array([p.x, p.y, p.z]) * cfg["transform"]["scale"]
      + cfg["transform"]["translation"]
    )
    quaternion = multiply(cfg["transform"]["quaternion_wxyz"], [q.w, q.x, q.y, q.z])
    result = PoseStamped()
    result.header.stamp = msg.header.stamp
    result.header.frame_id = cfg["world_frame"]
    result.pose.position.x, result.pose.position.y, result.pose.position.z = map(
      float, position
    )
    (
      result.pose.orientation.w,
      result.pose.orientation.x,
      result.pose.orientation.y,
      result.pose.orientation.z,
    ) = map(float, quaternion)
    return result

  def ball_callback(msg):
    transformed = transform(msg)
    point = PointStamped()
    point.header = transformed.header
    point.point.x = transformed.pose.position.x
    point.point.y = transformed.pose.position.y
    point.point.z = transformed.pose.position.z
    ball_pub.publish(point)
    poses = PoseArray()
    poses.header = transformed.header
    poses.poses = [transformed.pose]
    if root_pose[0] is not None:
      poses.poses.append(root_pose[0])
    poses_pub.publish(poses)

  def root_callback(msg):
    transformed = transform(msg)
    root_pose[0] = transformed.pose
    root_pub.publish(transformed)

  node.create_subscription(
    PoseStamped, cfg["vrpn_ball_topic"], ball_callback, qos_profile_sensor_data
  )
  node.create_subscription(
    PoseStamped, cfg["vrpn_root_topic"], root_callback, qos_profile_sensor_data
  )
  print(
    f"VRPN {cfg['vrpn_ball_topic']} -> {cfg['ball_topic']}, frame={cfg['world_frame']}"
  )
  try:
    rclpy.spin(node)
  finally:
    node.destroy_node()
    rclpy.shutdown()


def replay(config, csv_path):
  """Publish HOPE-style t,x,y,z recordings to the same live ball interface."""
  import rclpy
  from geometry_msgs.msg import PointStamped
  from rclpy.qos import qos_profile_sensor_data

  cfg, _ = read_config(config)
  with Path(csv_path).open() as stream:
    rows = list(csv.DictReader(stream))
  samples = np.array([[float(row[k]) for k in ("t", "x", "y", "z")] for row in rows])
  if (
    len(samples) < 3
    or not np.isfinite(samples).all()
    or np.any(np.diff(samples[:, 0]) <= 0)
  ):
    raise ValueError("CSV must contain increasing finite t,x,y,z samples")
  rclpy.init()
  node = rclpy.create_node("g1_pingpong_fake_mocap")
  pub = node.create_publisher(PointStamped, cfg["ball_topic"], qos_profile_sensor_data)
  start = time.monotonic()
  epoch = node.get_clock().now().nanoseconds
  try:
    for t, x, y, z in samples:
      offset = t - samples[0, 0]
      time.sleep(max(start + offset - time.monotonic(), 0))
      msg = PointStamped()
      ns = epoch + round(offset * 1e9)
      msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
      msg.header.frame_id = cfg["world_frame"]
      msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
      pub.publish(msg)
  finally:
    node.destroy_node()
    rclpy.shutdown()


def bridge(config):
  """Forward HOPE ROS position topics to the policy environment over UDP."""
  import rclpy
  from rclpy.executors import ExternalShutdownException

  cfg, _ = read_config(config)
  rclpy.init()
  node = rclpy.create_node("g1_pingpong_mocap_udp_bridge")
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  destination = (cfg["udp"]["host"], cfg["udp"]["port"])

  def send(packet):
    sock.sendto(json.dumps(packet, allow_nan=False).encode(), destination)

  subscribe_positions(node, cfg, send)
  if cfg.get("planner_command_topic"):
    subscribe_planner_commands(node, cfg, send)
  try:
    planner_topic = cfg.get("planner_command_topic")
    print(
      f"HOPE ROS topics -> UDP {destination}"
      + (f"; planner command input={planner_topic}" if planner_topic else "")
    )
    rclpy.spin(node)
  except (KeyboardInterrupt, ExternalShutdownException):
    pass
  finally:
    sock.close()
    node.destroy_node()
    # Humble's signal handler may have already shut down the ROS context.
    rclpy.try_shutdown()


def sim_bridge(config):
  """MuJoCo stand-in for vrpn_to_pose + bridge around an external ROS planner.

  Simulated normalized packets arrive over UDP from deploy (UDPPlannerLink) and
  are published as the robot's converter does: source.ball_topic (PoseArray,
  ball at slot 0, then root), source.root_topic and ball_topic (PointStamped),
  stamped with the simulated source time. /racket/command is relayed back over
  UDP exactly as in `bridge`.
  """
  import rclpy
  from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
  from rclpy.executors import ExternalShutdownException
  from rclpy.qos import qos_profile_sensor_data

  cfg, r = read_config(config)
  source, link = cfg["source"], cfg["simulation_planner_link"]
  if source.get("ball_type") != "pose_array" or source.get("ball_pose_index", 0) != 0:
    raise ValueError("sim-bridge publishes the ball at PoseArray slot 0")
  rclpy.init()
  node = rclpy.create_node("g1_pingpong_sim_planner_bridge")
  poses_pub = node.create_publisher(
    PoseArray, source["ball_topic"], qos_profile_sensor_data
  )
  root_pub = node.create_publisher(
    PoseStamped, source["root_topic"], qos_profile_sensor_data
  )
  point_pub = node.create_publisher(
    PointStamped, cfg["ball_topic"], qos_profile_sensor_data
  )
  inbound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  inbound.bind((link["ball_udp"]["host"], link["ball_udp"]["port"]))
  inbound.settimeout(0.1)
  outbound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  destination = (link["command_udp"]["host"], link["command_udp"]["port"])
  stop = threading.Event()
  root_pose = [None]
  transform = cfg["transform"]
  scale, translation = transform["scale"], np.asarray(transform["translation"])
  q = transform["quaternion_wxyz"]
  inverse = np.r_[q[0], -np.asarray(q[1:])]  # Conjugate of a unit quaternion.

  def to_source(packet):
    """Undo input_packet's normalization: deployment world -> source frame."""
    pose = PoseStamped().pose
    position = r.T @ (np.asarray(packet["position"]) - translation) / scale
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    if "quaternion_wxyz" in packet:
      q = multiply(inverse, packet["quaternion_wxyz"])
      (
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
      ) = map(float, q)
    return pose

  def header(packet, frame):
    msg = PoseStamped().header
    msg.stamp.sec, nanosec = divmod(round(packet["stamp"] * 1e9), 10**9)
    msg.stamp.nanosec = nanosec
    msg.frame_id = frame
    return msg

  def publish(packet):
    pose = to_source(packet)
    if packet["kind"] == "root":
      root_pose[0] = pose
      msg = PoseStamped()
      msg.header, msg.pose = header(packet, source["root_frame"]), pose
      root_pub.publish(msg)
      return
    poses = PoseArray()
    poses.header = header(packet, source["ball_frame"])
    poses.poses = [pose] + ([root_pose[0]] if root_pose[0] is not None else [])
    poses_pub.publish(poses)
    point = PointStamped()
    point.header = poses.header
    point.point.x, point.point.y, point.point.z = (
      pose.position.x,
      pose.position.y,
      pose.position.z,
    )
    point_pub.publish(point)

  def receive():
    while not stop.is_set():
      try:
        payload, _ = inbound.recvfrom(4096)
        publish(json.loads(payload))
      except TimeoutError:
        continue
      except (ValueError, KeyError, TypeError) as exc:
        node.get_logger().error(str(exc))

  def send(packet):
    outbound.sendto(json.dumps(packet, allow_nan=False).encode(), destination)

  subscribe_planner_commands(node, cfg, send)
  thread = threading.Thread(target=receive, daemon=True)
  thread.start()
  print(
    f"MuJoCo UDP {link['ball_udp']} -> {source['ball_topic']}, {source['root_topic']}, "
    f"{cfg['ball_topic']}; {cfg.get('planner_command_topic', '/racket/command')} "
    f"-> UDP {destination}",
    flush=True,
  )
  try:
    rclpy.spin(node)
  except (KeyboardInterrupt, ExternalShutdownException):
    pass
  finally:
    stop.set()
    thread.join(timeout=1)
    inbound.close()
    outbound.close()
    node.destroy_node()
    rclpy.try_shutdown()


def check_planner_physics(planner_config, rally_config):
  """Print whether an external planner's ball model matches the MuJoCo ball."""
  from deploy.motion_20261119.reference.hope_g1.hope_planner_adapter import ball_model_mismatch

  # Each planner package keys its parameters by its own node name.
  (node, parameters), = yaml.safe_load(Path(planner_config).read_text()).items()
  ball = yaml.safe_load(Path(rally_config).read_text())["ball"]
  mismatch = ball_model_mismatch(ball, parameters["ros__parameters"])
  if mismatch:
    print(
      f"WARNING: {node} ball model differs from the MuJoCo ball "
      f"{{key: [simulated, planner]}} {mismatch}; its predictions will be biased",
      flush=True,
    )
  else:
    print(f"{node} ball model matches the MuJoCo ball", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "mode", choices=["relay", "replay", "bridge", "sim-bridge", "check-planner-physics"]
  )
  parser.add_argument("--config", type=Path, default=Path("deploy/configs/mocap.yaml"))
  parser.add_argument("--csv", type=Path)
  parser.add_argument("--planner-config", type=Path)
  parser.add_argument("--rally-config", type=Path)
  args = parser.parse_args()
  if args.mode == "check-planner-physics":
    check_planner_physics(args.planner_config, args.rally_config)
  elif args.mode == "relay":
    relay(args.config)
  elif args.mode == "bridge":
    bridge(args.config)
  elif args.mode == "sim-bridge":
    sim_bridge(args.config)
  elif args.csv is not None:
    replay(args.config, args.csv)
  else:
    parser.error("replay requires --csv")


if __name__ == "__main__":
  main()

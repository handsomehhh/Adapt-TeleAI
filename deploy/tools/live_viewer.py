"""Real-time MuJoCo visualization of live mocap and planner data.

Subscribes to ROS topics (/poses, /racket/command) and displays:
- Robot joint state from mocap
- Ball trajectory
- Planner racket command visualization

This is a viewer only — no physics simulation, no control loop.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from deploy.backends import mujoco as mujoco_backend
from deploy.io import mocap, transport


class LiveState:
  """Thread-safe container for the latest mocap and planner data."""

  def __init__(self):
    self.lock = threading.Lock()
    self.robot_q = None
    self.robot_root_pos = None
    self.robot_root_quat = None
    self.ball_positions = deque(maxlen=100)  # Keep last 100 ball samples
    self.planner_command = None
    self.last_update = time.monotonic()

  def update_robot(self, q, root_pos, root_quat):
    with self.lock:
      self.robot_q = np.asarray(q, dtype=np.float64)
      self.robot_root_pos = np.asarray(root_pos, dtype=np.float64)
      self.robot_root_quat = np.asarray(root_quat, dtype=np.float64)
      self.last_update = time.monotonic()

  def update_ball(self, position):
    with self.lock:
      self.ball_positions.append((time.monotonic(), np.asarray(position, dtype=np.float64)))

  def update_planner(self, command):
    with self.lock:
      self.planner_command = command

  def snapshot(self):
    with self.lock:
      return dict(
        robot_q=self.robot_q.copy() if self.robot_q is not None else None,
        robot_root_pos=self.robot_root_pos.copy() if self.robot_root_pos is not None else None,
        robot_root_quat=self.robot_root_quat.copy() if self.robot_root_quat is not None else None,
        ball_positions=list(self.ball_positions),
        planner_command=dict(self.planner_command) if self.planner_command else None,
        age=time.monotonic() - self.last_update,
      )


def ros_receiver_thread(cfg, state, stop_event):
  """Background thread that receives ROS messages and updates state."""
  import rclpy
  from rclpy.node import Node
  from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

  rclpy.init()
  node = Node("hope_live_viewer")

  # Subscribe to mocap poses
  mocap_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
  )

  def poses_callback(msg):
    # msg is geometry_msgs/PoseArray
    # Index 0 is ball, index 1 is robot pelvis (P1)
    if len(msg.poses) >= 2:
      ball_pose = msg.poses[0]
      ball_pos = np.array([ball_pose.position.x, ball_pose.position.y, ball_pose.position.z])
      state.update_ball(ball_pos)

      robot_pose = msg.poses[1]
      root_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
      root_quat = np.array([
        robot_pose.orientation.w,
        robot_pose.orientation.x,
        robot_pose.orientation.y,
        robot_pose.orientation.z,
      ])
      # We don't have joint angles from /poses, will need /robot_state or similar
      # For now just update root
      state.update_robot(None, root_pos, root_quat)

  from geometry_msgs.msg import PoseArray
  node.create_subscription(PoseArray, "/poses", poses_callback, mocap_qos)

  # Subscribe to planner command
  def planner_callback(msg):
    command = dict(
      position=np.array([msg.position.x, msg.position.y, msg.position.z]),
      velocity=np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z]),
      normal=np.array([msg.normal.x, msg.normal.y, msg.normal.z]),
      time_to_strike=float(msg.time_to_strike),
      valid=bool(msg.valid),
    )
    state.update_planner(command)

  from hope_msgs.msg import RacketCommand
  planner_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
  )
  node.create_subscription(RacketCommand, "/racket/command", planner_callback, planner_qos)

  # Spin until stop requested
  while not stop_event.is_set():
    rclpy.spin_once(node, timeout_sec=0.1)

  node.destroy_node()
  rclpy.shutdown()


def add_visual_elements(scene, data, state_snapshot, ball_config):
  """Add ball trajectory and planner command to the MuJoCo scene."""
  # Draw ball positions as small spheres
  now = time.monotonic()
  for t, pos in state_snapshot["ball_positions"]:
    age = now - t
    if age > 2.0:  # Only show last 2 seconds
      continue
    alpha = max(0.1, 1.0 - age / 2.0)
    mujoco.mjv_initGeom(
      scene.geoms[scene.ngeom],
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.zeros(3),
      np.zeros(3),
      np.zeros(9),
      np.array([1.0, 0.5, 0.0, alpha]),  # Orange, fading
    )
    scene.geoms[scene.ngeom].pos[:] = pos
    scene.geoms[scene.ngeom].size[0] = ball_config.get("ball_radius", 0.02)
    scene.ngeom = min(scene.ngeom + 1, scene.maxgeom)

  # Draw planner command as a frame at target position
  cmd = state_snapshot["planner_command"]
  if cmd and cmd.get("valid"):
    pos = cmd["position"]
    vel = cmd["velocity"]
    normal = cmd["normal"]

    # Draw target position as a sphere
    mujoco.mjv_initGeom(
      scene.geoms[scene.ngeom],
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.zeros(3),
      np.zeros(3),
      np.zeros(9),
      np.array([0.0, 1.0, 0.0, 0.8]),  # Green
    )
    scene.geoms[scene.ngeom].pos[:] = pos
    scene.geoms[scene.ngeom].size[0] = 0.03
    scene.ngeom = min(scene.ngeom + 1, scene.maxgeom)

    # Draw velocity as an arrow
    if np.linalg.norm(vel) > 0.01:
      arrow_end = pos + 0.3 * vel / np.linalg.norm(vel)
      mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([0.0, 0.8, 0.0, 0.6]),
      )
      scene.geoms[scene.ngeom].pos[:] = pos
      # Arrow from pos to arrow_end
      direction = arrow_end - pos
      scene.geoms[scene.ngeom].mat[:3] = direction / np.linalg.norm(direction)
      scene.geoms[scene.ngeom].mat[3:6] = np.array([0, 0, 1])  # Up vector
      scene.geoms[scene.ngeom].mat[6:] = np.cross(scene.geoms[scene.ngeom].mat[:3], [0, 0, 1])
      scene.geoms[scene.ngeom].size[0] = 0.01  # Width
      scene.geoms[scene.ngeom].size[1] = 0.02  # Head width
      scene.geoms[scene.ngeom].size[2] = np.linalg.norm(direction)  # Length
      scene.ngeom = min(scene.ngeom + 1, scene.maxgeom)

    # Draw normal as a short arrow
    if np.linalg.norm(normal) > 0.01:
      normal_end = pos + 0.1 * normal
      mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([1.0, 1.0, 0.0, 0.6]),  # Yellow
      )
      scene.geoms[scene.ngeom].pos[:] = pos
      direction = normal_end - pos
      scene.geoms[scene.ngeom].mat[:3] = direction / np.linalg.norm(direction)
      scene.geoms[scene.ngeom].mat[3:6] = np.array([0, 0, 1])
      scene.geoms[scene.ngeom].mat[6:] = np.cross(scene.geoms[scene.ngeom].mat[:3], [0, 0, 1])
      scene.geoms[scene.ngeom].size[0] = 0.008
      scene.geoms[scene.ngeom].size[1] = 0.015
      scene.geoms[scene.ngeom].size[2] = np.linalg.norm(direction)
      scene.ngeom = min(scene.ngeom + 1, scene.maxgeom)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--real-config",
    type=Path,
    default=Path("deploy/hope_g1_hitter/configs/real.yaml"),
    help="Robot configuration file",
  )
  parser.add_argument(
    "--rally-config",
    type=Path,
    default=Path("deploy/hope_g1_hitter/configs/rally.yaml"),
    help="Rally configuration with ball parameters",
  )
  parser.add_argument("--fps", type=int, default=30, help="Viewer refresh rate")
  args = parser.parse_args()

  import yaml
  real_cfg = yaml.safe_load(args.real_config.read_text())
  rally_cfg = yaml.safe_load(args.rally_config.read_text())

  # Load MuJoCo model
  robot_cfg = real_cfg["robot"]
  model_path = Path(robot_cfg["mujoco_model_path"])
  if not model_path.is_absolute():
    model_path = Path(__file__).parent.parent / model_path
  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)

  # Apply world offset if specified
  world_offset = np.array(rally_cfg.get("world", {}).get("mujoco_to_world_translation", [0, 0, 0]))
  if np.any(world_offset != 0):
    # Shift the floor/table geoms
    for i in range(model.ngeom):
      if model.geom(i).name in ("floor", "table", "table_top"):
        model.geom_pos[i] += world_offset

  # Initialize viewer
  viewer = mujoco.viewer.launch_passive(model, data)
  viewer.cam.distance = 3.0
  viewer.cam.azimuth = 135
  viewer.cam.elevation = -20

  # Start ROS receiver thread
  state = LiveState()
  stop_event = threading.Event()
  receiver = threading.Thread(target=ros_receiver_thread, args=(rally_cfg, state, stop_event), daemon=True)
  receiver.start()

  print("Live viewer started. Listening to /poses and /racket/command...")
  print("Press Ctrl+C to exit.")

  frame_dt = 1.0 / args.fps

  try:
    while viewer.is_running():
      frame_start = time.monotonic()

      # Get latest state
      snapshot = state.snapshot()

      # Update robot pose in MuJoCo data
      if snapshot["robot_root_pos"] is not None and snapshot["robot_root_quat"] is not None:
        data.qpos[:3] = snapshot["robot_root_pos"] + world_offset
        data.qpos[3:7] = snapshot["robot_root_quat"]

      if snapshot["robot_q"] is not None and len(snapshot["robot_q"]) == model.nq - 7:
        data.qpos[7:] = snapshot["robot_q"]

      # Forward kinematics (no physics, just update transforms)
      mujoco.mj_forward(model, data)

      # Render scene
      viewer.sync()

      # Add visual overlays (ball trail, planner command)
      # Note: mjv_scene is not directly accessible in passive viewer
      # We'll need to use the viewer's scene

      # Sleep to maintain frame rate
      elapsed = time.monotonic() - frame_start
      if elapsed < frame_dt:
        time.sleep(frame_dt - elapsed)

      # Display data age warning
      if snapshot["age"] > 1.0:
        print(f"Warning: No mocap data for {snapshot['age']:.1f}s", end="\r")

  except KeyboardInterrupt:
    print("\nShutting down...")
  finally:
    stop_event.set()
    receiver.join(timeout=2.0)
    viewer.close()


if __name__ == "__main__":
  main()

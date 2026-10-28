"""World-frame task command shared by planners, selectors and benchmarks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RacketCommand:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  deadline: float
  diagnostic: dict


def decode_ros_planner_command(packet, received):
  """Decode the local ROS bridge packet using the real HOPE command contract.

  Return (command, error); an invalid packet must clear any previous command.
  The bridge and controller share the host monotonic clock, so UDP latency
  must not extend the strike deadline.
  """
  if not packet.get("valid", False):
    return None, "ros_planner_published_invalid_command"
  try:
    position = np.asarray(packet["position"], dtype=float)
    velocity = np.asarray(packet["velocity"], dtype=float)
    normal = np.asarray(packet["normal"], dtype=float)
    tts = float(packet["time_to_strike"])
    published = float(packet.get("bridge_monotonic", received))
  except (KeyError, TypeError, ValueError):
    return None, "malformed_ros_planner_command"
  if (
    position.shape != (3,)
    or velocity.shape != (3,)
    or normal.shape != (3,)
    or not np.isfinite(position).all()
    or not np.isfinite(velocity).all()
    or not np.isfinite(normal).all()
    or np.linalg.norm(normal) < 1e-6
    or not np.isfinite(tts)
    or not np.isfinite(received)
    or not np.isfinite(published)
    or tts <= 0
  ):
    return None, "malformed_ros_planner_command"
  diagnostic = {
    "source": "hope_planner_optimized_2_A30808_ros",
    "source_stamp": float(packet.get("source_ball_time", 0.0)),
    "planner_stamp": float(packet.get("stamp", 0.0)),
    "received_monotonic": received,
    "time_to_strike": tts,
    "position": position.tolist(),
    "velocity": velocity.tolist(),
    "normal": normal.tolist(),
    "outgoing_velocity": packet.get("outgoing_velocity"),
    "valid": True,
    "clears_net": bool(packet.get("clears_net", False)),
    "bypasses_net_posts": bool(packet.get("bypasses_net_posts", False)),
    "num_bounces": int(packet.get("predicted_bounces", 0)),
  }
  return RacketCommand(
    position=position,
    velocity=velocity,
    normal=normal,
    deadline=published + tts,
    diagnostic=diagnostic,
  ), ""

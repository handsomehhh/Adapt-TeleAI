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


def decode_g1_planner_command(
  packet, received, *, contract_hash, session_id, runtime_mode=None, physics_sha256=None
):
  """Consume a complete G1 solve without re-solving station or clipping velocity."""
  if runtime_mode is not None and packet.get("runtime_mode") != runtime_mode:
    return None, "wrong_runtime_mode"
  if physics_sha256 is not None and packet.get("physics_sha256") != physics_sha256:
    return None, "wrong_physics_profile"
  if packet.get("schema") != "g1_1v1_v1":
    return None, "wrong_planner_schema"
  if packet.get("contract_sha256") != contract_hash:
    return None, "wrong_policy_contract"
  if packet.get("session_id") != session_id:
    return None, "previous_policy_session"
  try:
    station = np.asarray(packet["station"], dtype=float)
    side = int(packet["side"])
    if station.shape != (2,) or not np.isfinite(station).all() or side not in (0, 1):
      return None, "invalid_g1_station_or_side"
    command, error = decode_ros_planner_command(packet, received)
    if command is None:
      return None, error
    command.diagnostic.update(
      source="hope_planner_g1",
      station=station.tolist(),
      side=side,
      command_id=int(packet["command_id"]),
      state_sequence=int(packet["state_sequence"]),
      outgoing_velocity_requested=packet.get("outgoing_velocity_requested"),
      outgoing_velocity_residual=packet.get("outgoing_velocity_residual"),
    )
    return command, ""
  except (ValueError, TypeError, KeyError):
    return None, "malformed_g1_command"

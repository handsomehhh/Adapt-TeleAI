"""HOPE table world and the explicit MuJoCo plant-to-mocap bridge.

The policy/real/planner world has its origin at the near-left tabletop corner:
table X=[0, 2.74], Y=[-1.525, 0], Z=0. MuJoCo retains a floor at Z=0.
Only simulated measurements and incoming reference geometry cross this bridge;
joint commands, actor outputs, velocities and orientations are not modified.
"""

import numpy as np


def simulation_translation(world):
  if world.get("name") != "hope_table":
    raise ValueError("HOPE G1 requires world.name=hope_table")
  value = np.asarray(world["mujoco_to_world_translation"], dtype=float)
  if value.shape != (3,) or not np.isfinite(value).all():
    raise ValueError("Invalid MuJoCo-to-HOPE translation")
  return value


def mujoco_ball_config(ball, translation):
  """Describe the same physical court in the simulator's floor world."""
  offset = np.asarray(translation)
  result = dict(ball)
  for key in ("table_near_x", "table_far_x"):
    result[key] -= float(offset[0])
  result["table_center_y"] = ball.get("table_center_y", 0.0) - float(offset[1])
  result["table_height"] -= float(offset[2])
  result["target_xy"] = (np.asarray(ball["target_xy"]) - offset[:2]).tolist()
  if "target_land_z" in ball:
    result["target_land_z"] -= float(offset[2])
  result["coordinate_frame"] = "mujoco_floor"
  if not np.isclose(result["table_center_y"], 0):
    raise ValueError("MuJoCo analytical court must be centred on Y=0")
  return result


def validate_table_world(ball):
  if not (
    np.isclose(ball["table_near_x"], 0)
    and np.isclose(ball["table_far_x"], 2.74)
    and np.isclose(ball.get("table_center_y", 0), -0.7625)
    and np.isclose(ball["table_width"], 1.525)
    and np.isclose(ball["table_height"], 0)
  ):
    raise ValueError("HOPE table must be X=[0,2.74], Y=[-1.525,0], Z=0")

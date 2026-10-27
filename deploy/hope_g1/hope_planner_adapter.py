"""In-process adapter for the latest A30808 ROS HOPE planner core.

Only the ROS transport is replaced. Every source ball packet is passed to the
same HOPEPlanner implementation used by the ROS node, and its absolute source
timestamp is mapped to the deployment monotonic clock at packet receipt.
"""

from pathlib import Path
import queue
import threading

import numpy as np
import yaml

from deploy.hope_g1.hope_planner_optimized_2.constants import (
  BallPhysics,
  PlannerConfig,
  TableParams,
)
from deploy.hope_g1.hope_planner_optimized_2.planner import HOPEPlanner
from deploy.racket_command import RacketCommand


class HopePlannerAdapter:
  def __init__(self, hit_plane_x=None, config_path=None):
    if config_path is None:
      config_path = Path(__file__).with_name("configs") / "hope_planner_optimized_2.yaml"
    raw = yaml.safe_load(Path(config_path).read_text())
    params = raw["hope_planner_optimized_2"]["ros__parameters"]
    self.params = params
    self.hit_plane_x = float(params["x_hit"] if hit_plane_x is None else hit_plane_x)
    publish_x_min = float(params["publish_intercept_x_min"])
    publish_x_max = float(params["publish_intercept_x_max"])
    if hit_plane_x is not None:
      publish_x_min = publish_x_max = self.hit_plane_x
    config = PlannerConfig(
      poly_order=2,
      fit_window=30,
      min_fit_samples=10,
      mocap_hz=180.0,
      clear_x_max=2.2,
      x_hit=self.hit_plane_x,
      target_land=np.array([params["target_land_x"], params["target_land_y"], 0.0]),
      delta_t_flight=float(params["delta_t_flight"]),
      C_r=float(params["restitution_racket"]),
      racket_radius=0.075,
      dt_integrate=float(params["dt_integrate"]),
      max_predict_time=2.0,
      bounce_z_tol=0.02,
      stage3_dt_integrate=float(params["stage3_dt_integrate"]),
      outgoing_velocity_tolerance_m=float(params["outgoing_velocity_tolerance_m"]),
      time_to_strike_invalid_x=float(params["time_to_strike_invalid_x"]),
      publish_intercept_x_min=publish_x_min,
      publish_intercept_x_max=publish_x_max,
      publish_intercept_y_min=float(params["publish_intercept_y_min"]),
      publish_intercept_y_max=float(params["publish_intercept_y_max"]),
      publish_intercept_z_min=float(params["publish_intercept_z_min"]),
      publish_intercept_z_max=float(params["publish_intercept_z_max"]),
    )
    physics = BallPhysics(
      k=float(params["drag_k"]),
      C_h=float(params["restitution_h"]),
      C_v=float(params["restitution_v"]),
      g=np.array([0.0, 0.0, -9.81]),
      radius=0.02,
      mass=0.0027,
    )
    table = TableParams(
      length=2.74,
      width=1.525,
      height=0.76,
      net_x=1.37,
      net_height=0.1525,
      net_overhang=0.15,
    )
    self.planner = HOPEPlanner(physics, config, table)
    self.latest = None
    self.last_update_diagnostics = {}
    self._incoming = queue.Queue(maxsize=8192)
    self._lock = threading.Lock()
    self.dropped_observations = 0
    self._worker = threading.Thread(target=self._process, daemon=True)
    self._worker.start()

  def update_observations(self, observations):
    """Queue raw ball packets in callback order without blocking policy ticks."""
    for stamp, position, received in observations:
      try:
        self._incoming.put_nowait((float(stamp), np.asarray(position, dtype=float).copy(), float(received)))
      except queue.Full:
        self.dropped_observations += 1

  def _process(self):
    while True:
      stamp, position, received = self._incoming.get()
      command = self.planner.update(float(stamp), np.asarray(position, dtype=float))
      diagnostics = dict(self.planner.last_update_diagnostics)
      with self._lock:
        self.last_update_diagnostics = diagnostics
      if command is None:
        continue
      remaining = float(command.t_strike) - float(stamp)
      latest = RacketCommand(
        position=np.asarray(command.p_intercept, dtype=float).copy(),
        velocity=np.asarray(command.v_racket, dtype=float).copy(),
        normal=np.asarray(command.n_racket, dtype=float).copy(),
        deadline=float(received) + remaining,
        diagnostic={
          "source": "hope_planner_optimized_2_A30808",
          "source_stamp": float(stamp),
          "source_ball_position": np.asarray(position, dtype=float).tolist(),
          "received_monotonic": float(received),
          "time_to_strike": remaining,
          "hit_plane_x": self.hit_plane_x,
          "position": np.asarray(command.p_intercept).tolist(),
          "velocity": np.asarray(command.v_racket).tolist(),
          "normal": np.asarray(command.n_racket).tolist(),
          "outgoing_velocity": np.asarray(command.v_ball_outgoing).tolist(),
          "target_land": np.asarray(command.target_land).tolist(),
          "clears_net": bool(command.clears_net),
          "bypasses_net_posts": bool(command.bypasses_net_posts),
          "valid": bool(command.valid),
          "num_bounces": int(command.num_bounces),
          "stage1_fit_rmse_m": float(command.stage1_fit_rmse_m),
        },
      )
      with self._lock:
        self.latest = latest

  def latest_command(self, now):
    with self._lock:
      if self.latest is None or self.latest.deadline <= now:
        return None
      return self.latest

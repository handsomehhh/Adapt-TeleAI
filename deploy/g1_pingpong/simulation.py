"""Legacy simplified rally evaluator, retained for historical tests.

The deployment CLI full/standing modes use simulation_runtime.run_deployment,
which shares run_real including PD, AMP and handoff. This evaluator does not.
"""

import copy
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from deploy.backends.mujoco import MujocoBackend
from deploy.backends.playback import HELP as PLAYBACK_HELP
from deploy.backends.playback import PlaybackControl
from deploy.g1_pingpong.commands import CommandGuard, policy_pd_settings
from deploy.g1_pingpong.core import rotation
from deploy.g1_pingpong.feeder import RandomBallFeeder
from deploy.g1_pingpong.oscillation import OscillationGuard
from deploy.g1_pingpong.perception import BallEstimator
from deploy.g1_pingpong.rally import RallyController
from deploy.io.mocap import read_config
from deploy.io.transport import build_mocap


def record_tick(stream, engine, state, now, target, ball, failure, tilt):
  """Keep post-halt physics visible without reusing stale policy outputs."""
  if stream is None:
    return
  halted = engine.lifecycle.state == "HALT"
  row = {
    "time": now,
    "state": engine.lifecycle.state,
    "phase": engine.phase,
    "control_mode": "damping" if halted else "active",
    "motion_index": engine.motion_index,
    "completed": engine.completed,
    "recovered": engine.recovered,
    "q": state.q.tolist(),
    "dq": state.dq.tolist(),
    "root_pos": state.root_pos.tolist(),
    "root_quat": state.root_quat.tolist(),
    "racket_position": state.racket_pos.tolist(),
    "racket_normal": state.racket_normal.tolist(),
    "tilt_degrees": tilt,
    "target_q": target.tolist() if target is not None else None,
    "ball_position": ball.position.tolist() if ball is not None else None,
    "failure": failure,
  }
  if not halted:
    row.update(
      {
        ("policy_target_q" if k == "target_q" else k): v.tolist()
        if isinstance(v, np.ndarray)
        else v
        for k, v in engine.debug.items()
      }
    )
  stream.write(json.dumps(row) + "\n")


def run_rally(
  deployment,
  real_config,
  mocap_config,
  *,
  simulation_config="deploy/configs/mujoco.yaml",
  motion_index=5,
  episodes=10,
  duration=120,
  render=False,
  seed=42,
  output=None,
  standing_only=False,
  motion_select="auto",
  motion_allow=None,
  ball_source="sim",
  final_hold_seconds=1.0,
):
  cfg = deployment.settings
  if cfg["stage"] != 2:
    raise ValueError("Receiving rally requires a Stage 2 checkpoint")
  if not np.isfinite(final_hold_seconds) or final_hold_seconds <= 0:
    raise ValueError("final_hold_seconds must be finite and positive")
  runtime = yaml.safe_load(Path(real_config).read_text())
  trained_robot_contract = cfg["robot"]
  cfg = policy_pd_settings(cfg, runtime)
  options = yaml.safe_load(Path(simulation_config).read_text())
  sensor, _ = read_config(mocap_config)
  robot = MujocoBackend(copy.copy(deployment.model), cfg)
  engine = RallyController(
    deployment,
    runtime,
    options,
    home_index=motion_index,
    allow=motion_allow,
    selection=motion_select,
    receiving=not standing_only,
  )
  print(engine.describe(), flush=True)
  feeder = RandomBallFeeder(
    cfg["ball"],
    options["feed"],
    np.random.default_rng(seed),
  )
  guard = CommandGuard(
    cfg["robot"]["joint_limits"], runtime["guard"], cfg["robot"]["joint_names"]
  )
  oscillation = OscillationGuard(
    cfg["robot"]["joint_names"],
    runtime["guard"].get("oscillation", {}),
    cfg["control_dt"],
  )
  estimator = BallEstimator(**sensor["estimator"])
  result = {
    "mode": "standing" if standing_only else "rally",
    "robot_command_contract": cfg["robot"],
    "trained_robot_command_contract": trained_robot_contract,
    "checkpoint": str(deployment.checkpoint),
    "home_motion": motion_index,
    "motion_select": motion_select,
    "motion_allow": engine.candidate_indices,
    "requested_swings": episodes,
    "final_hold_seconds": final_hold_seconds,
    "robot_resets": 1,
    "robot_commands_sent": False,
    "phases": [],
    "events": engine.events,
    "swings": [],
    "failure": None,
    "passed": False,
    "max_tilt_degrees": 0.0,
    "max_control_tilt_degrees": 0.0,
    "post_halt_max_tilt_degrees": None,
  }
  ball = viewer = stream = provider = None
  last_sensor = -1.0
  next_serve = None
  swing_ball = None
  complete_since = None
  if output:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / "trace.jsonl").open("w")
  playback = PlaybackControl()
  playback.reset(0.)
  dt = cfg["control_dt"]
  now = 0.0
  try:
    if ball_source == "mocap":
      provider = build_mocap(sensor)
    if render:
      from deploy import viewer as mujoco_viewer

      viewer = mujoco_viewer.launch_passive(robot.model, robot.data, key_callback=playback.key)
      print(f"MuJoCo playback: {PLAYBACK_HELP}", flush=True)
    for step in range(int(np.ceil(duration / dt))):
      now = step * dt
      if result["failure"]:
        if viewer is None or not viewer.is_running():
          break
        state = robot.state()
        tilt = float(
          np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
        )
        result["max_tilt_degrees"] = max(result["max_tilt_degrees"], tilt)
        result["post_halt_max_tilt_degrees"] = max(
          result["post_halt_max_tilt_degrees"] or 0.0, tilt
        )
        record_tick(stream, engine, state, now, None, ball, result["failure"], tilt)
        for _ in range(cfg["decimation"]):
          robot.substep()
        viewer.sync()
        playback.wait((step+1)*dt)
        continue
      state = robot.state()
      if provider is not None:
        estimate = provider.ball.get(time.monotonic())
        if estimate is not None:
          estimate.received = now - max(0, time.monotonic() - estimate.received)
      else:
        estimate = estimator.get(now)
      if episodes > 0 and engine.completed >= episodes:
        engine.receiving = False
      target = None
      try:
        before = engine.lifecycle.state
        target = engine.step(state, estimate, now)
        oscillation.check(state.dq, now)
        guard.check(target, state.q, state.dq, dt)
        robot.set_target(target)
        if engine.lifecycle.state == "ARMED" and provider is None:
          if next_serve is None:
            next_serve = now + options["feed"]["serve_gap_seconds"]
          if ball is None and now >= next_serve:
            ball = feeder.sample(engine.candidates(state))
            ball.previous_racket = state.racket_pos.copy()
            estimator = BallEstimator(**sensor["estimator"])
            result["events"].append(
              {
                "time": now,
                "event": "serve_created",
                "position": ball.position.tolist(),
                "velocity": ball.velocity.tolist(),
              }
            )
          if ball is not None and ball.elapsed > ball.arrival + 1.0:
            result["events"].append({"time": now, "event": "serve_skipped"})
            ball = None
            estimator = BallEstimator(**sensor["estimator"])
            next_serve = now + options["feed"]["serve_gap_seconds"]
        elif engine.lifecycle.state == "EXECUTE":
          swing_ball = ball
          next_serve = None
        if before == "EXECUTE" and engine.lifecycle.state == "RECOVER":
          result["swings"].append(
            {
              "time": now,
              "motion_index": engine.motion_index,
              "motion": engine.library.paths[engine.motion_index].name,
              "hit_type": engine.library.hit_types[engine.motion_index],
              "hit": bool(swing_ball and swing_ball.hit),
              "landed": bool(swing_ball and swing_ball.landed),
              "success": bool(swing_ball and getattr(swing_ball, "success", False)),
            }
          )
        if before == "RECOVER" and engine.lifecycle.state == "READY":
          ball = None
          estimator = BallEstimator(**sensor["estimator"])
          next_serve = None
        for substep in range(cfg["decimation"]):
          robot.substep()
          sensor_time = now + (substep + 1) * cfg["physics_dt"]
          if ball is not None:
            ball.substep(robot.state(), cfg["physics_dt"])
            if sensor_time - last_sensor >= 0.01 - 1e-9:
              estimator.update(ball.position, sensor_time, received=sensor_time)
              last_sensor = sensor_time
          if engine.lifecycle.state == "RECOVER" and result["swings"] and swing_ball:
            result["swings"][-1].update(
              hit=bool(swing_ball.hit),
              landed=bool(swing_ball.landed),
              success=bool(getattr(swing_ball, "success", False)),
            )
      except (ValueError, TimeoutError, RuntimeError) as error:
        result["failure"] = {
          "time": now,
          "phase": engine.lifecycle.state,
          "motion_index": engine.motion_index,
          "motion": engine.library.paths[engine.motion_index].name,
          "policy_phase": engine.debug.get("policy_phase"),
          "reference_time": engine.debug.get("reference_time"),
          "root_pos": state.root_pos.tolist(),
          "tilt_degrees": float(
            np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
          ),
          "requested_target_q": target.tolist() if target is not None else None,
          "error": str(error),
        }
        engine.lifecycle.halt()
        robot.damp(runtime["halt_kd"])
        print(json.dumps(result["failure"]), flush=True)
        target = None
      tilt = float(
        np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
      )
      result["max_tilt_degrees"] = max(result["max_tilt_degrees"], tilt)
      result["max_control_tilt_degrees"] = max(result["max_control_tilt_degrees"], tilt)
      if not result["phases"] or result["phases"][-1] != engine.lifecycle.state:
        result["phases"].append(engine.lifecycle.state)
        print(
          f"[{now:.2f}s] {engine.lifecycle.state} motion={engine.motion_index} completed={engine.completed}",
          flush=True,
        )
      record_tick(stream, engine, state, now, target, ball, result["failure"], tilt)
      if viewer:
        if not viewer.is_running():
          result["termination"] = "viewer_closed"
          break
        with viewer.lock():
          viewer.user_scn.ngeom = 0
          for index, reference in engine.candidates(state):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            selected = index == engine.motion_index
            mujoco.mjv_initGeom(
              geom,
              mujoco.mjtGeom.mjGEOM_SPHERE,
              np.full(3, 0.025 if selected else 0.012),
              reference.strike_position(cfg["ball"]),
              np.eye(3).ravel(),
              np.array([1.0, 0.85, 0.1, 0.8])
              if selected
              else np.array([0.2, 0.8, 1.0, 0.5]),
            )
            geom.label = f"m{index}"
            viewer.user_scn.ngeom += 1
          if ball is not None:
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
              geom,
              mujoco.mjtGeom.mjGEOM_SPHERE,
              np.full(3, cfg["ball"]["ball_radius"]),
              ball.position,
              np.eye(3).ravel(),
              np.array([1.0, 0.65, 0.1, 1.0]),
            )
            viewer.user_scn.ngeom += 1
        viewer.sync()
        playback.wait((step+1)*dt)
      elif provider is not None:
        playback.wait((step+1)*dt)
      if result["failure"]:
        if not render:
          break
      if (
        episodes > 0
        and engine.recovered >= episodes
        and engine.lifecycle.state == "READY"
      ):
        if complete_since is None:
          complete_since = now
        if now - complete_since >= final_hold_seconds:
          result["passed"] = True
          result["termination"] = "all_swings_recovered"
          break
      else:
        complete_since = None
    if standing_only and engine.lifecycle.state == "READY" and now + dt >= duration:
      result["passed"] = True
      result["termination"] = "standing_hold_completed"
    result.update(
      completed_swings=engine.completed,
      recovered_swings=engine.recovered,
      state=engine.lifecycle.state,
      phase=engine.phase,
      simulation_time=float(robot.data.time),
    )
    result["termination"] = result.get(
      "termination", "halt" if result["failure"] else "duration"
    )
  except KeyboardInterrupt:
    result["termination"] = "interrupted"
  finally:
    if provider:
      provider.close()
    if stream:
      stream.close()
    if viewer:
      viewer.close()
    if output:
      (output / "summary.json").write_text(json.dumps(result, indent=2))
  print(json.dumps(result), flush=True)
  return result

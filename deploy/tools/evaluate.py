"""Independent reset-episode evaluation; separate from continuous deployment."""

import json
from pathlib import Path

import numpy as np

from deploy.backends.mujoco import MujocoBackend, SimBall
from deploy.backends.playback import HELP as PLAYBACK_HELP
from deploy.backends.playback import PlaybackControl
from deploy.g1_pingpong.core import Controller, Motion, rotation
from deploy.tools.traces import input_snapshot, save_trace


def rollout_summary(results):
  count = len(results)
  summary = {
    "episodes": count,
    "robot_failure_rate": sum(row["robot_failure"] for row in results) / count
    if count
    else 0.0,
  }
  if results and "hit" in results[0]:
    for field in ("hit", "net_clear", "success"):
      summary[field + "_rate"] = sum(row[field] for row in results) / count
    errors = [
      row["landing_error"]
      for row in results
      if row["success"] and row["landing_error"] is not None
    ]
    summary["landing_error_mean_given_success"] = (
      float(np.mean(errors)) if errors else None
    )
    summary["successful_landings"] = len(errors)
  return summary


def run_mujoco(
  deployment,
  *,
  motion_index=0,
  episodes=1,
  all_motions=False,
  render=False,
  seed=42,
  output=None,
  fixed_speed=None,
  zero_residual=False,
):
  cfg = deployment.settings
  tracker = deployment.tracker
  actor = deployment.actor
  backend = MujocoBackend(deployment.model, cfg)
  rng = np.random.default_rng(seed)
  indices = list(range(len(cfg["motions"]))) if all_motions else [motion_index]
  results, viewer = [], None
  playback = PlaybackControl()
  if render:
    from deploy import viewer as deployment_viewer

    viewer = deployment_viewer.launch_passive(backend.model, backend.data, key_callback=playback.key)
    print(f"MuJoCo playback: {PLAYBACK_HELP}", flush=True)
  try:
    for index in indices:
      motion = Motion(
        deployment.motions[index],
        cfg["robot"]["joint_names"],
        cfg["anchor"],
      )
      for repeat in range(episodes):
        controller = Controller(cfg, motion, tracker, actor)
        backend.reset(motion)
        ball = SimBall(cfg["ball"], motion, rng) if actor else None
        if ball:
          ball.previous_racket = backend.state().racket_pos
        traces, failed, reason = [], False, "time_out"
        playback.reset(0.)
        thresholds = cfg.get("termination_thresholds", {})
        for step in range(
          round(cfg.get("episode_length_s", 3 if actor else 4) / cfg["control_dt"])
        ):
          state = backend.state()
          ball_state = ball.observation() if ball else None
          snapshot = input_snapshot(controller, state, ball_state)
          target = controller.step(
            state,
            ball_state,
            fixed_speed=fixed_speed,
            zero_residual=zero_residual,
          )
          traces.append({**snapshot, **controller.debug})
          backend.set_target(target)
          for _ in range(cfg["decimation"]):
            backend.substep()
            if ball:
              ball.substep(backend.state(), cfg["physics_dt"])
          state = backend.state()
          t = controller.debug["reference_time"]
          _, ref_pos, ref_quat = motion.sample(t)
          gravity = np.array([0.0, 0.0, -1.0])
          orientation_error = abs(
            (rotation(state.anchor_quat).T @ gravity)[2]
            - (rotation(ref_quat).T @ gravity)[2]
          )
          feet_z = [
            motion.body_position(n, t)[2]
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")
          ]
          feet_error = np.max(np.abs(backend.data.xpos[backend.feet, 2] - feet_z))
          if (
            abs(state.anchor_pos[2] - ref_pos[2])
            > thresholds.get("anchor_height", 0.25)
            or orientation_error > thresholds.get("anchor_orientation", 0.8)
            or feet_error > thresholds.get("feet_height", 0.25)
          ):
            failed, reason = True, "robot_failure"
            break
          if ball and (ball.landed or ball.missed):
            reason = "ball_finished"
            break
          if not ball and controller.debug[
            "reference_time"
          ] >= motion.duration + cfg.get("warmdown_s", 0.25):
            reason = "motion_finished"
            break
          if viewer:
            if not viewer.is_running():
              return results
            with viewer.lock():
              if ball:
                viewer.user_scn.ngeom = 1
                mujoco.mjv_initGeom(
                  viewer.user_scn.geoms[0],
                  mujoco.mjtGeom.mjGEOM_SPHERE,
                  np.array([cfg["ball"]["ball_radius"]] * 3),
                  ball.position,
                  np.eye(3).reshape(-1),
                  np.array([1.0, 0.65, 0.1, 1.0]),
                )
            viewer.sync()
            playback.wait((step+1)*cfg["control_dt"])
        report = {
          "motion": cfg["motions"][index],
          "repeat": repeat,
          "steps": len(traces),
          "robot_failure": failed,
          "termination": reason,
          "completion": min(controller.t / motion.duration, 1.0),
        }
        if ball:
          report.update(
            hit=ball.hit,
            net_clear=ball.net_clear,
            success=getattr(ball, "success", False),
            landing_error=ball.landing_error,
          )
        results.append(report)
        print(json.dumps(report))
        if output:
          output = Path(output)
          output.mkdir(parents=True, exist_ok=True)
          save_trace(
            output / f"trace_{index}_{repeat}.npz",
            traces,
            cfg["motions"][index],
            cfg["stage"],
          )
          (output / "rollouts.json").write_text(json.dumps(results, indent=2))
          (output / "summary.json").write_text(
            json.dumps(rollout_summary(results), indent=2)
          )
  finally:
    if viewer:
      viewer.close()
  return results

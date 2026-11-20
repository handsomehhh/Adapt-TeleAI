"""Evaluate trained G1 tracker and MVAE without changing training artifacts.

Run with PYTHONPATH pointing to the original training snapshot. This evaluates
known tracker clips, the original held-out MVAE clips, and unplanned generated
motions. It does not measure the learned rally planner or real hardware.
"""

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_JOINT_VELOCITY_LIMITS,
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_RACKET_NORMAL,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_adapt.data import load_clip
from mjlab.tasks.g1_adapt.generator import MotionGenerator
from mjlab.tasks.g1_adapt.motion import blend_state
from mjlab.tasks.g1_adapt.mvae import load_mvae
from mjlab.tasks.g1_adapt.reference import GeneratedMotionCommandCfg
from mjlab.tasks.g1_stage1.env_cfg import g1_stage1_env_cfg
from mjlab.tasks.g1_stage1.mdp.tracking import Stage1MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
  path = Path(path)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def distribution(values):
  values = np.asarray(values, dtype=float)
  return {
    "mean": float(values.mean()),
    "p95": float(np.percentile(values, 95)),
    "max": float(values.max()),
  }


def state_errors(model, prediction, target):
  return {
    "normalized_state_mse": float(
      (model.normalize(prediction) - model.normalize(target)).square().mean()
    ),
    "joint_rmse_rad": float(
      (prediction[..., 12:41] - target[..., 12:41]).square().mean().sqrt()
    ),
    "root_velocity_rmse_m_s": float(
      (prediction[..., :3] - target[..., :3]).square().mean().sqrt()
    ),
    "joint_velocity_rmse_rad_s": float(
      (prediction[..., 41:70] - target[..., 41:70]).square().mean().sqrt()
    ),
  }


def offline_mvae(model, metadata, device):
  report = {}
  for split, key in (("training", "train_clips"), ("validation", "validation_clips")):
    clips = [load_clip(item["path"]) for item in metadata[key]]
    pairs = torch.cat([c["state"].unfold(0, 2, 1).transpose(1, 2) for c in clips])
    previous, target = pairs[:, 0].to(device), pairs[:, 1].to(device)
    history = previous[:, None]
    encoded = model.encoder(
      torch.cat((model.normalize(previous), model.normalize(target)), -1)
    )
    mean, logvar = model.posterior(encoded).chunk(2, -1)
    posterior = model.decode(history, mean)[0]
    prior = model.decode(history, torch.zeros_like(mean))[0]
    result = {
      "clips": len(clips),
      "transitions": len(pairs),
      "posterior_mean_target_informed": state_errors(model, posterior, target),
      "prior_zero_latent": state_errors(model, prior, target),
      "persistence_baseline": state_errors(model, previous, target),
      "posterior_latent_std_across_samples": float(mean.std(0).mean()),
      "mean_posterior_sigma": float((0.5 * logvar.clamp(-8, 4)).exp().mean()),
    }
    windows = torch.cat([c["state"].unfold(0, 9, 4).transpose(1, 2) for c in clips]).to(
      device
    )
    result["rollout_8_windows"] = len(windows)
    result["rollout_8"] = {}
    for mode in ("zero", "normal_0.5", "normal_1.0", "persistence"):
      torch.manual_seed(20261103)
      history = windows[:, :1]
      predictions = []
      for _ in range(8):
        if mode == "persistence":
          predicted = windows[:, 0]
        else:
          latent = torch.zeros(len(windows), model.cfg.latent_dim, device=device)
          if mode != "zero":
            latent = torch.randn_like(latent) * float(mode.split("_")[-1])
          proposed = model.decode(history, latent.clamp(-3, 3))[0]
          predicted = blend_state(
            history[:, -1], proposed, torch.ones(len(windows), device=device)
          )
        predictions.append(predicted)
        history = predicted[:, None]
      predictions = torch.stack(predictions, 1)
      result["rollout_8"][mode] = state_errors(model, predictions, windows[:, 1:])
      result["rollout_8"][mode]["per_step_normalized_mse"] = [
        float(
          (model.normalize(predictions[:, t]) - model.normalize(windows[:, t + 1]))
          .square()
          .mean()
        )
        for t in range(8)
      ]
    report[split] = result
    print("Offline MVAE:", split, json.dumps(result), flush=True)
  return report


def free_generation(model, metadata, device, output):
  paths = [Path(c["path"]) for c in metadata["validation_clips"]]
  states = torch.stack([load_clip(p)["state"][0] for p in paths]).to(device)
  roots = []
  for path in paths:
    with np.load(path) as data:
      root_id = list(data["body_names"]).index("pelvis")
      roots.append(data["body_pos_w"][0, root_id])
  roots = torch.as_tensor(np.stack(roots), device=device)
  limits = torch.tensor(
    [G1_PINGPONG_JOINT_VELOCITY_LIMITS[n] for n in G1_PINGPONG_JOINT_NAMES],
    device=device,
  )
  report, traces = {}, {}
  for mode in ("zero", "normal_0.5", "normal_1.0"):
    torch.manual_seed(20261103)
    generator = MotionGenerator(model, len(paths), device)
    generator.reset(torch.arange(len(paths), device=device), states, roots)
    generated, roots_log, saturation, velocity_violations = [], [], [], []
    inconsistencies, minimum_feet = [], []
    feet = [
      generator.fk.names.index(n)
      for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    raw_limits = []
    first_nonfinite_step = None
    for step in range(250):
      latent = torch.zeros(len(paths), model.cfg.latent_dim, device=device)
      if mode != "zero":
        latent = torch.randn_like(latent) * float(mode.split("_")[-1])
      latent = latent.clamp(-3, 3)
      proposed = model.decode(generator.history, latent)[0]
      if not torch.isfinite(proposed).all():
        first_nonfinite_step = step + 1
        break
      raw_limits.append(
        (
          (proposed[:, 12:41] < generator.fk.limits[:, 0])
          | (proposed[:, 12:41] > generator.fk.limits[:, 1])
        )
        .float()
        .mean()
        .item()
      )
      state = generator.step(latent, torch.ones(len(paths), device=device))
      generated.append(state.cpu())
      roots_log.append(generator.root.cpu())
      saturation.append(
        (
          (state[:, 12:41] <= generator.fk.limits[:, 0] + 1e-5)
          | (state[:, 12:41] >= generator.fk.limits[:, 1] - 1e-5)
        )
        .float()
        .mean()
        .item()
      )
      velocity_violations.append(
        (generator.joint_velocities.abs() > limits).float().mean().item()
      )
      inconsistencies.append(
        (generator.joint_velocities - state[:, 41:70]).square().mean().sqrt().item()
      )
      minimum_feet.append(generator.positions[:, feet, 2].min(-1).values.cpu())
    if not generated:
      report[mode] = {"first_nonfinite_step": first_nonfinite_step}
      continue
    sequence, positions = torch.stack(generated), torch.stack(roots_log)
    phase = torch.atan2(sequence[..., 71], sequence[..., 70]).numpy()
    phase_progress = np.unwrap(phase, axis=0)[-1] - phase[0]
    report[mode] = {
      "steps": len(generated),
      "duration_s": len(generated) * 0.02,
      "first_nonfinite_step": first_nonfinite_step,
      "raw_joint_limit_violation_fraction": float(np.mean(raw_limits)),
      "joint_saturation_fraction": float(np.mean(saturation)),
      "finite_difference_velocity_limit_violation_fraction": float(
        np.mean(velocity_violations)
      ),
      "state_vs_finite_difference_qdot_rmse_rad_s": float(np.mean(inconsistencies)),
      "max_root_displacement_m": float((positions - roots.cpu()).norm(dim=-1).max()),
      "root_height_range_m": [
        float(positions[..., 2].min()),
        float(positions[..., 2].max()),
      ],
      "min_ankle_height_m": float(torch.stack(minimum_feet).min()),
      "ankle_below_ground_fraction": float(
        (torch.stack(minimum_feet) < 0).float().mean()
      ),
      "mean_joint_temporal_std_rad": float(sequence[..., 12:41].std(0).mean()),
      "phase_progress_cycles_per_clip": (phase_progress / (2 * np.pi)).tolist(),
      "phase_total_variation_rad": float(
        np.abs(np.diff(np.unwrap(phase, axis=0), axis=0)).mean(1).sum()
      ),
    }
    traces[f"{mode}_states"] = sequence.numpy()
    traces[f"{mode}_roots"] = positions.numpy()
    print("Free generation:", mode, json.dumps(report[mode]), flush=True)
  np.savez_compressed(output / "mvae_free_generation.npz", **traces)
  return report


def make_env(paths, device, mvae=None):
  cfg = g1_stage1_env_cfg(play=True)
  cfg.scene.num_envs = len(paths)
  cfg.terminations = {}
  cfg.rewards = {}
  # Keep noise objects at construction; the manager drops them permanently
  # when corruption is disabled. Individual trials toggle these objects below.
  cfg.observations["actor"].enable_corruption = True
  motion = cast(Stage1MotionCommandCfg, cfg.commands["motion"])
  motion.motion_files = tuple(map(str, paths))
  motion.motion_directory = ""
  motion.debug_vis = False
  if mvae:
    motion = GeneratedMotionCommandCfg(
      **{f.name: getattr(motion, f.name) for f in fields(motion)}
    )
    motion.mvae_file = str(mvae)
    cfg.commands["motion"] = motion
  return ManagerBasedRlEnv(cfg, device=device)


def reset_exact(env, seed, generated):
  env.reset(seed=seed)
  command = env.command_manager.get_term("motion")
  ids = torch.arange(env.num_envs, device=env.device)
  command.motion_ids[:] = ids
  # Generated references must temporarily be disabled when writing raw seed states.
  generator = getattr(command, "generator", None)
  if generated:
    command.generator = None
  command.reset_to_frame(ids, 0)
  if generated:
    command.generator = generator
    command._initialize_generated(ids)
  env.sim.forward()
  command.update_relative_body_poses()


def measure(env):
  command = env.command_manager.get_term("motion")
  data = env.scene["robot"].data
  wrist = command.cfg.body_names.index("right_wrist_yaw_link")
  actual_wrist = env.scene["robot"].body_names.index("right_wrist_yaw_link")
  ref_q = command.body_quat_w[:, wrist]
  actual_q = data.body_link_quat_w[:, actual_wrist]
  offset = torch.tensor(G1_PINGPONG_MOUNT_OFFSET, device=env.device).expand(
    env.num_envs, -1
  )
  normal = torch.tensor(G1_PINGPONG_RACKET_NORMAL, device=env.device).expand(
    env.num_envs, -1
  )
  reference_face = command.body_pos_w[:, wrist] + quat_apply(ref_q, offset)
  actual_face = data.body_link_pos_w[:, actual_wrist] + quat_apply(actual_q, offset)
  qerror = data.joint_pos[:, command._motion_to_robot_joint_ids] - command.joint_pos
  reference_normal, actual_normal = (
    quat_apply(ref_q, normal),
    quat_apply(actual_q, normal),
  )
  normal_angle = torch.acos((reference_normal * actual_normal).sum(-1).clamp(-1, 1)) * (
    180 / np.pi
  )
  values = {
    "joint_rms_rad": qerror.square().mean(-1).sqrt(),
    "wrist_error_m": (
      data.body_link_pos_w[:, actual_wrist] - command.body_pos_w[:, wrist]
    ).norm(dim=-1),
    "racket_center_error_m": (actual_face - reference_face).norm(dim=-1),
    "racket_normal_error_deg": normal_angle,
    "height_error_m": (data.root_link_pos_w[:, 2] - command.body_pos_w[:, 0, 2]).abs(),
    "gravity_z": data.projected_gravity_b[:, 2],
    "root_height_m": data.root_link_pos_w[:, 2],
  }
  finite = torch.stack([torch.isfinite(v) for v in values.values()]).all(0)
  values["gate_failed"] = (
    ~finite
    | (values["height_error_m"] > 0.25)
    | (values["gravity_z"] > -0.65)
    | (values["joint_rms_rad"] > 0.5)
    | (values["wrist_error_m"] > 0.3)
  )
  values["fallen"] = (
    ~finite | (values["root_height_m"] < 0.45) | (values["gravity_z"] > -0.5)
  )
  return {key: value.cpu().numpy() for key, value in values.items()}


def physics_trials(paths, tracker, device, output, mvae=None, hold_seconds=0.0):
  env = make_env(paths, device, mvae)
  generated = mvae is not None
  trials, summary = [], []
  noise_specs = {
    name: env.observation_manager.get_term_cfg("actor", name).noise
    for name in env.observation_manager.active_terms["actor"]
  }
  try:
    conditions: list[tuple[str, float | None, bool, int]] = [
      (str(speed), speed, False, 20261103) for speed in (0.5, 1.0, 1.5, 2.0)
    ]
    if not generated:
      conditions.append(("random", None, False, 20261103))
      conditions += [
        (str(speed), speed, True, seed)
        for seed in (20261104, 20261105, 20261106)
        for speed in (0.5, 1.0, 1.5, 2.0)
      ]
      conditions += [
        ("random", None, True, seed) for seed in (20261104, 20261105, 20261106)
      ]
    else:
      conditions = [
        (mode, speed, False, 20261103)
        for mode in ("zero", "normal_0.5", "normal_1.0")
        for speed in (0.5, 1.0, 2.0)
      ]
    if hold_seconds:
      conditions = [(str(speed), speed, False, 20261103) for speed in (0.5, 1.0, 2.0)]
    for label, speed, noise, seed in conditions:
      for name, spec in noise_specs.items():
        env.observation_manager.get_term_cfg("actor", name).noise = (
          spec if noise else None
        )
      reset_exact(env, seed, generated)
      command = env.command_manager.get_term("motion")
      end = (command.motion.lengths - 1) / command.motion.fps_values
      active = torch.ones(len(paths), dtype=torch.bool, device=device)
      values, masks = [], []
      first_fall = np.full(len(paths), np.nan)
      max_steps = (
        250 if generated else int(float(end.max()) / (0.02 * (speed or 0.5))) + 2
      )
      if hold_seconds:
        max_steps = round(hold_seconds / env.step_dt)
      for step in range(max_steps):
        if not generated and not active.any():
          break
        if generated:
          assert speed is not None
          latent = torch.zeros(len(paths), 32, device=device)
          if label != "zero":
            latent = torch.randn_like(latent) * float(label.split("_")[-1])
          command.advance(
            latent.clamp(-3, 3), torch.full((len(paths),), speed, device=device)
          )
        observation = env.observation_manager.compute_group("actor")
        action = tracker(observation)
        if not torch.isfinite(action).all():
          raise RuntimeError("Non-finite tracker action")
        if not generated:
          advance = (
            torch.full((len(paths),), speed, device=device)
            if speed
            else torch.rand(len(paths), device=device) * 1.5 + 0.5
          )
          command.set_pending_dt(advance * env.step_dt)
        env.step(action)
        row = measure(env)
        mask = active.cpu().numpy().copy()
        newly_fallen = row["fallen"] & mask & np.isnan(first_fall)
        first_fall[newly_fallen] = (step + 1) * env.step_dt
        values.append(row)
        masks.append(mask)
        if not generated and not hold_seconds:
          active &= command.current_time_seconds() < end - 1e-6
      masks = np.stack(masks)
      arrays = {key: np.stack([row[key] for row in values]) for key in values[0]}
      for index, path in enumerate(paths):
        valid = masks[:, index]
        metrics = {
          key: distribution(arr[valid, index])
          for key, arr in arrays.items()
          if key not in ("fallen", "gate_failed")
        }
        trials.append(
          {
            "clip": path.name,
            "condition": label,
            "speed": speed,
            "observation_noise": noise,
            "seed": seed,
            "steps": int(valid.sum()),
            "passed_correction_gate": not bool(
              arrays["gate_failed"][valid, index].any()
            ),
            "fallen": bool(arrays["fallen"][valid, index].any()),
            "first_fall_s": None
            if np.isnan(first_fall[index])
            else float(first_fall[index]),
            "metrics": metrics,
          }
        )
      current = trials[-len(paths) :]
      item = {
        "condition": label,
        "speed": speed,
        "observation_noise": noise,
        "seed": seed,
        "clips": len(paths),
        "passed_gate": sum(t["passed_correction_gate"] for t in current),
        "falls": sum(t["fallen"] for t in current),
        "joint_rms_rad": distribution(arrays["joint_rms_rad"][masks]),
        "wrist_error_m": distribution(arrays["wrist_error_m"][masks]),
        "racket_center_error_m": distribution(arrays["racket_center_error_m"][masks]),
        "racket_normal_error_deg": distribution(
          arrays["racket_normal_error_deg"][masks]
        ),
      }
      summary.append(item)
      save(
        output
        / (
          "generated_tracking.json"
          if generated
          else "replay_hold_tracking.json"
          if hold_seconds
          else "tracker.json"
        ),
        {"summary": summary, "trials": trials},
      )
      print(
        "Generated tracker" if generated else "Tracker", json.dumps(item), flush=True
      )
    return {"summary": summary, "trials": trials}
  finally:
    env.close()
    torch.cuda.empty_cache()


def strike_trials(paths, tracker, device, output):
  """Evaluate the annotated strike instant and a +/-60ms reference-time window."""
  env = make_env(paths, device)
  report, traces = [], {}
  strikes = []
  for path in paths:
    with np.load(path) as data:
      frame = float(np.asarray(data["strike_frame"]).reshape(-1)[0])
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      strikes.append(frame / fps)
  try:
    for speed in (0.5, 1.0, 1.5, 2.0):
      for name in env.observation_manager.active_terms["actor"]:
        env.observation_manager.get_term_cfg("actor", name).noise = None
      reset_exact(env, 20261103, False)
      command = env.command_manager.get_term("motion")
      times, rows = [], []
      steps = int((max(strikes) + 0.1) / (speed * env.step_dt)) + 1
      for _ in range(steps):
        action = tracker(env.observation_manager.compute_group("actor"))
        command.set_pending_dt(
          torch.full((len(paths),), speed * env.step_dt, device=device)
        )
        env.step(action)
        times.append(command.current_time_seconds().cpu().numpy().copy())
        rows.append(measure(env))
      times = np.stack(times)
      arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
      window = abs(times - np.asarray(strikes)[None]) <= 0.060001
      nearest = abs(times - np.asarray(strikes)[None]).argmin(0)
      index = np.arange(len(paths))
      item = {"speed": speed, "clips": len(paths), "reference_time_window_s": 0.06}
      item["at_nearest_strike_frame"] = {
        key: distribution(arr[nearest, index])
        for key, arr in arrays.items()
        if key not in ("fallen", "gate_failed")
      }
      item["strike_window"] = {
        key: distribution(arr[window])
        for key, arr in arrays.items()
        if key not in ("fallen", "gate_failed")
      }
      report.append(item)
      for key in ("wrist_error_m", "racket_center_error_m", "racket_normal_error_deg"):
        traces[f"speed_{speed}_{key}"] = arrays[key]
      traces[f"speed_{speed}_time_s"] = times
      print("Strike window:", json.dumps(item), flush=True)
    save(output / "tracker_strike.json", report)
    np.savez_compressed(output / "tracker_strike_traces.npz", **traces)
    return report
  finally:
    env.close()
    torch.cuda.empty_cache()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  run, output = args.run.resolve(), args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  status = json.loads((run / "status.json").read_text())
  tracker_pt = Path(status["stages"]["stage1"]["checkpoint"])
  tracker_path = (
    tracker_pt.parent / "jit" / f"modeljit_{tracker_pt.stem.split('_')[-1]}.pt"
  )
  mvae_path = run / "mvae/mvae.pt"
  provenance = json.loads((run / "provenance.json").read_text())
  for relative, expected in provenance["source_files"].items():
    if digest(run / "snapshot" / relative) != expected:
      raise ValueError(f"Source snapshot changed: {relative}")
  report = {
    "run": str(run),
    "device": args.device,
    "tracker_checkpoint": str(tracker_path),
    "tracker_sha256": digest(tracker_path),
    "mvae": str(mvae_path),
    "mvae_sha256": digest(mvae_path),
    "evaluation_script_sha256": digest(__file__),
    "scope": "Known raw tracker clips; original MVAE clip-level holdout; generated references without a planner. No ball or table in component physics tests.",
    "correction_gate": {
      "height_error_m": 0.25,
      "gravity_z_max": -0.65,
      "joint_rms_rad": 0.5,
      "wrist_error_m": 0.3,
    },
  }
  save(output / "report.json", report)
  torch.set_num_threads(4)
  with torch.inference_mode():
    model, metadata = load_mvae(mvae_path, args.device)
    report["mvae_offline"] = offline_mvae(model, metadata, args.device)
    save(output / "report.json", report)
    report["mvae_free_generation"] = free_generation(
      model, metadata, args.device, output
    )
    save(output / "report.json", report)
    tracker = torch.jit.load(str(tracker_path), map_location=args.device).eval()
    paths = sorted((run / "snapshot/dataset/g1_stage1/motions").glob("*.npz"))
    report["tracker"] = physics_trials(paths, tracker, args.device, output)
    report["tracker_strike"] = strike_trials(paths, tracker, args.device, output)
    save(output / "report.json", report)
    validation = [Path(c["path"]) for c in metadata["validation_clips"]]
    report["replay_hold_tracking"] = physics_trials(
      validation, tracker, args.device, output, hold_seconds=5.0
    )
    report["generated_tracking"] = physics_trials(
      validation, tracker, args.device, output, mvae_path
    )
    report["completed"] = True
    save(output / "report.json", report)
  print("Completed component validation:", output, flush=True)


if __name__ == "__main__":
  main()

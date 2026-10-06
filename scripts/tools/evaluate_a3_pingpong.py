"""Seeded frame-zero evaluation of A3 or G1 ping-pong checkpoints.

Example::

  uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
    --checkpoint LOG/model_0.pt LOG/model_999.pt --output LOG/validation/rollouts.json

Uses the JIT policy exported next to each checkpoint, without constructing an
optimizer. Stage 1 fixes reference speed to 1; Stage 2 retains learned timing.
Each clip is explicitly assigned to an environment, including clips the
training adaptive sampler may rarely select. Only the first episode counts.
Receiving ablations can freeze reference speed with ``--fixed-speed`` and/or
remove joint corrections with ``--zero-residual``. Incoming shots retain the
same seeded distribution for every checkpoint and ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.a3_pingpong.actions import FrozenTrackerActionCfg
from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommandCfg
from mjlab.tasks.a3_pingpong.runner import validate_tracker_dependency
from mjlab.tasks.adapt_tennis.mdp.commands import (
  MotionCommandCfg,
  _resolve_motion_paths,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-PingPong-Tracking-A3-Stage1-RandomDt"


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def ablate_receiving_actions(
  actions: torch.Tensor, *, fixed_speed: bool, zero_residual: bool
) -> torch.Tensor:
  """A zero final action means reference speed 1 in FrozenTrackerAction."""
  if not fixed_speed and not zero_residual:
    return actions
  actions = actions.clone()
  if zero_residual:
    actions[:, :-1] = 0
  if fixed_speed:
    actions[:, -1] = 0
  return actions


def receiving_failure_masks(
  reasons: dict[str, torch.Tensor], receiving: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
  """A ball_finished signal includes valid landings; it is not itself failure."""
  missed = receiving["ball/state_missed"].bool()
  robot_failed = torch.zeros_like(missed)
  for name in ("anchor_height", "anchor_orientation", "feet_height"):
    if name in reasons:
      robot_failed |= reasons[name]
  return robot_failed | missed, robot_failed


def receiving_landing_statistics(receiving: dict[str, torch.Tensor]) -> dict:
  """Exclude default zero landing errors of shots that never landed."""
  errors = receiving.get("ball/state_landing_error")
  if errors is None:
    return {}
  result = {}
  for state, label, denominator in (
    ("landed", "landing", "landings"),
    ("success", "success", "successes"),
  ):
    mask = receiving[f"ball/state_{state}"].bool()
    count = int(mask.sum())
    result[f"landing_error_mean_given_{label}_m"] = (
      float(errors[mask].mean()) if count else None
    )
    result[f"landing_error_denominator_{denominator}"] = count
  return result


class PolicyArtifacts(NamedTuple):
  """Resolved files behind one evaluator ``--checkpoint`` argument."""

  requested: Path
  checkpoint: Path | None
  jit: Path


def resolve_policy_artifacts(path: Path) -> PolicyArtifacts:
  """Resolve a run directory, standard checkpoint, or standalone JIT policy."""
  requested = path.expanduser()
  checkpoint: Path | None = None
  if requested.is_dir():
    candidates = [
      candidate
      for candidate in requested.glob("model_*.pt")
      if candidate.stem.removeprefix("model_").isdigit()
    ]
    if not candidates:
      raise FileNotFoundError(f"No model_*.pt in {requested}")
    checkpoint = max(
      candidates, key=lambda candidate: int(candidate.stem.removeprefix("model_"))
    ).resolve()
  elif requested.stem.startswith("model_"):
    checkpoint = requested.resolve()
    if not checkpoint.is_file():
      raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

  if checkpoint is not None:
    suffix = checkpoint.stem.removeprefix("model_")
    jit = checkpoint.parent / "jit" / f"modeljit_{suffix}.pt"
  else:
    jit = requested
  if not jit.is_file():
    raise FileNotFoundError(f"Exported JIT policy not found: {jit}")
  return PolicyArtifacts(requested=requested, checkpoint=checkpoint, jit=jit.resolve())


def resolve_jit(path: Path) -> Path:
  """Backward-compatible JIT resolver used by the strike diagnostic."""
  return resolve_policy_artifacts(path).jit


def checkpoint_tracker_sha256(checkpoint: Path | None) -> str | None:
  """Read the frozen-tracker dependency recorded by a standard checkpoint."""
  if checkpoint is None:
    return None
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict):
    raise ValueError(f"Checkpoint is not a mapping: {checkpoint}")
  infos = payload.get("infos") or {}
  if not isinstance(infos, dict):
    raise ValueError(f"Checkpoint infos is not a mapping: {checkpoint}")
  expected = infos.get("pingpong_tracker_sha256", infos.get("a3_tracker_sha256"))
  if expected is not None and not isinstance(expected, str):
    raise ValueError(f"Checkpoint tracker SHA256 is not a string: {checkpoint}")
  return expected


def validate_policy_tracker_dependency(
  artifacts: PolicyArtifacts, actual_tracker_sha256: str | None
) -> str | None:
  """Validate a Stage-2 checkpoint before loading its exported policy."""
  expected = checkpoint_tracker_sha256(artifacts.checkpoint)
  validate_tracker_dependency(expected, actual_tracker_sha256)
  return expected


def policy_provenance(
  artifacts: PolicyArtifacts, expected_tracker_sha256: str | None
) -> dict[str, str | bool | None]:
  """Describe the exact model checkpoint, JIT policy, and tracker contract."""
  return {
    "checkpoint": str(artifacts.requested),
    "resolved_checkpoint": str(artifacts.checkpoint)
    if artifacts.checkpoint is not None
    else None,
    "checkpoint_sha256": sha256(artifacts.checkpoint)
    if artifacts.checkpoint is not None
    else None,
    "checkpoint_tracker_sha256": expected_tracker_sha256,
    "tracker_dependency_validated": expected_tracker_sha256 is not None,
    "jit_policy": str(artifacts.jit),
    "jit_sha256": sha256(artifacts.jit),
  }


def reset_all_clips(env, seed: int):
  """Reset at exactly frame zero after overriding randomized motion selection."""
  env.reset(seed=seed)
  motion = env.command_manager.get_term("motion")
  ids = torch.arange(env.num_envs, device=env.device)
  motion.motion_ids[:] = ids % motion.motion.num_motions
  motion.reset_to_frame(ids, 0)
  motion.last_dt[:] = 0
  motion._pending_dt[:] = motion.cfg.fixed_dt
  if hasattr(motion, "_start_time"):
    motion._start_time[:] = 0
  env.action_manager.reset(ids)
  env.episode_length_buf[:] = 0
  env.scene.write_data_to_sim()
  env.sim.forward()
  motion.update_relative_body_poses()
  # Incoming ball targets must refer to the explicitly selected clips.
  for name in env.cfg.commands:
    if name != "motion":
      term = env.command_manager.get_term(name)
      term._resample_command(ids)
  env.sim.sense()
  env.obs_buf = env.observation_manager.compute(update_history=True)
  return motion, env.obs_buf


def snapshot_receiving(env) -> dict[str, torch.Tensor]:
  """Read only per-environment metrics; absent for plain tracking tasks."""
  values = {}
  for name in env.cfg.commands:
    if name == "motion":
      continue
    term = env.command_manager.get_term(name)
    if hasattr(term, "_update_metrics"):
      term._update_metrics()
    for metric_name, value in getattr(term, "metrics", {}).items():
      if isinstance(value, torch.Tensor) and value.shape == (env.num_envs,):
        values[f"{name}/{metric_name}"] = value.detach().clone()
    for field in (
      "hit",
      "net_cleared",
      "landed",
      "success",
      "missed",
      "min_racket_distance",
      "landing_error",
    ):
      value = getattr(term, field, None)
      if isinstance(value, torch.Tensor) and value.shape == (env.num_envs,):
        values[f"{name}/state_{field}"] = value.detach().clone().float()
  return values


@torch.inference_mode()
def evaluate_policy(
  env,
  policy,
  files: list[Path],
  seed: int,
  *,
  fixed_speed: bool = False,
  zero_residual: bool = False,
) -> dict:
  is_receiving = "ball" in env.cfg.commands
  if (fixed_speed or zero_residual) and not is_receiving:
    raise ValueError(
      "Reference-speed/residual ablations require a Stage-2 receiving task."
    )
  motion, observations = reset_all_clips(env, seed)
  count = env.num_envs
  device = env.device
  active = torch.ones(count, dtype=torch.bool, device=device)
  completed = torch.zeros_like(active)
  failed = torch.zeros_like(active)
  ended = torch.zeros_like(active)
  steps_alive = torch.zeros(count, dtype=torch.long, device=device)
  samples = torch.zeros(count, device=device)
  sums = {
    key: torch.zeros(count, device=device)
    for key in (
      "relative_body_position_error_m",
      "global_body_position_error_m",
      "joint_position_rmse_rad",
      "anchor_position_error_m",
    )
  }
  completion = torch.zeros(count, device=device)
  cumulative_return = torch.zeros(count, device=device)
  receiving = {}
  reasons = {
    name: torch.zeros_like(active) for name in env.termination_manager.active_terms
  }
  lengths = (motion.motion.lengths[motion.motion_ids] - 1) / motion.motion.fps_values[
    motion.motion_ids
  ]
  terminal_time = torch.zeros(count, device=device)
  terminal_speed = torch.ones(count, device=device)
  speed_sum = torch.zeros(count, device=device)
  speed_square_sum = torch.zeros(count, device=device)
  action_term = env.action_manager.get_term("joint_pos")
  original_reset = env._reset_idx

  # ManagerBasedRlEnv auto-resets before returning a step. Preserve terminal
  # command/ball state here so the report cannot accidentally read a new shot.
  def capture_terminal(env_ids=None):
    ids = env_ids if env_ids is not None else torch.arange(count, device=device)
    ids = ids[active[ids]]
    if len(ids):
      terminal_time[ids] = motion.current_time_seconds(ids)
      if is_receiving:
        terminal_speed[ids] = action_term.speed[ids]
      for name in reasons:
        reasons[name][ids] = env.termination_manager.get_term(name)[ids]
      for name, value in snapshot_receiving(env).items():
        if name not in receiving:
          receiving[name] = torch.zeros(count, device=device)
        receiving[name][ids] = value[ids]
    original_reset(env_ids)

  env._reset_idx = capture_terminal
  max_steps = int(math.ceil((env.cfg.episode_length_s + 1) / env.step_dt))
  try:
    with torch.inference_mode():
      for _ in range(max_steps):
        current_time = motion.current_time_seconds()
        completed |= active & (current_time >= lengths - 1e-5)
        completion[active] = (current_time[active] / lengths[active]).clamp(0, 1)
        joint_diff = (
          motion.joint_pos
          - motion.robot_joint_pos[:, motion._motion_to_robot_joint_ids]
        )
        metrics = {
          "relative_body_position_error_m": (
            motion.body_pos_relative_w - motion.robot_body_pos_w
          )
          .norm(dim=-1)
          .mean(-1),
          "global_body_position_error_m": (motion.body_pos_w - motion.robot_body_pos_w)
          .norm(dim=-1)
          .mean(-1),
          "joint_position_rmse_rad": joint_diff.square().mean(-1).sqrt(),
          "anchor_position_error_m": (
            motion.anchor_pos_w - motion.robot_anchor_pos_w
          ).norm(dim=-1),
        }
        for name, value in metrics.items():
          if not torch.isfinite(value[active]).all():
            raise RuntimeError(f"Non-finite evaluation metric {name}")
          sums[name][active] += value[active]
        samples[active] += 1
        actions = policy(observations["actor"])
        if is_receiving:
          actions = ablate_receiving_actions(
            actions, fixed_speed=fixed_speed, zero_residual=zero_residual
          )
        if not torch.isfinite(actions).all():
          raise RuntimeError("Policy produced non-finite actions.")
        observations, reward, terminated, truncated, _ = env.step(actions)
        steps_alive[active] += 1
        cumulative_return[active] += reward[active]
        done = active & (terminated | truncated)
        speed = (
          action_term.speed.clone()
          if is_receiving
          else torch.ones(count, device=device)
        )
        # The action term is reset during env.step; retain the action used for
        # the terminal physics interval instead of its reset value of 1.
        speed[done] = terminal_speed[done]
        speed_sum[active] += speed[active]
        speed_square_sum[active] += speed[active].square()
        if is_receiving and "ball/state_missed" in receiving:
          task_failures, _ = receiving_failure_masks(reasons, receiving)
          failed |= done & task_failures
        else:
          failed |= done & terminated
        ended |= done
        completion[done] = (terminal_time[done] / lengths[done]).clamp(0, 1)
        completed |= done & ~terminated & (completion >= 1 - 1e-5)
        active &= ~done
        if not active.any():
          break
  finally:
    env._reset_idx = original_reset
  if active.any():
    for name, value in snapshot_receiving(env).items():
      if name not in receiving:
        receiving[name] = torch.zeros(count, device=device)
      receiving[name][active] = value[active]
  mean_metrics = {name: value / samples.clamp_min(1) for name, value in sums.items()}
  speed_mean = speed_sum / steps_alive.clamp_min(1)
  speed_std = (
    (speed_square_sum / steps_alive.clamp_min(1) - speed_mean.square())
    .clamp_min(0)
    .sqrt()
  )
  rows = []
  for i in range(count):
    with np.load(files[i % len(files)], allow_pickle=False) as archive:
      hit_type = str(archive["hit_type"]) if "hit_type" in archive else "unknown"
    rows.append(
      {
        "motion": files[i % len(files)].name,
        "hit_type": hit_type,
        "repeat": i // len(files),
        "completed_full_reference_clip": bool(completed[i]),
        "failed_termination": bool(failed[i]),
        "episode_ended": bool(ended[i]),
        "steps": int(steps_alive[i]),
        "elapsed_s": float(steps_alive[i]) * env.step_dt,
        "completion_fraction": float(completion[i]),
        "return": float(cumulative_return[i]),
        "reference_speed_mean": float(speed_mean[i]),
        "reference_speed_std": float(speed_std[i]),
        "termination_reasons": [
          name for name, value in reasons.items() if bool(value[i])
        ],
        **{name: float(value[i]) for name, value in mean_metrics.items()},
        "receiving_metrics_at_episode_end": {
          name: float(value[i]) for name, value in receiving.items()
        },
      }
    )
  summary = {
    "episodes": count,
    "full_clip_completion_rate": float(completed.float().mean()),
    "completed_episode_without_failure_rate": float(
      (ended & completed & ~failed).float().mean()
    ),
    "failure_termination_rate": float(failed.float().mean()),
    "mean_completion_fraction": float(completion.mean()),
    "mean_episode_steps": float(steps_alive.float().mean()),
    "mean_episode_seconds": float(steps_alive.float().mean()) * env.step_dt,
    "mean_return": float(cumulative_return.mean()),
    "reference_speed_mean": float(speed_sum.sum() / steps_alive.sum().clamp_min(1)),
    "reference_speed_std": float(
      (
        speed_square_sum.sum() / steps_alive.sum().clamp_min(1)
        - (speed_sum.sum() / steps_alive.sum().clamp_min(1)).square()
      )
      .clamp_min(0)
      .sqrt()
    ),
    "unfinished_at_evaluation_limit": int(active.sum()),
    **{name: float(value.mean()) for name, value in mean_metrics.items()},
    "termination_rates": {
      name: float(value.float().mean()) for name, value in reasons.items()
    },
    "receiving_metrics_at_episode_end": {
      name: float(value.mean()) for name, value in receiving.items()
    },
  }
  hit = receiving.get("ball/state_hit")
  success = receiving.get("ball/state_success")
  if hit is not None and success is not None:
    summary["return_success_given_racket_hit"] = (
      float(success[hit.bool()].mean()) if bool(hit.any()) else None
    )
    summary["return_success_denominator_hits"] = int(hit.sum())
    _, robot_failed = receiving_failure_masks(reasons, receiving)
    summary["robot_failure_rate"] = float(robot_failed.float().mean())
    summary["ball_return_success_rate"] = float(success.mean())
    summary["receiving_task_success_rate"] = float(
      (success.bool() & ~robot_failed).float().mean()
    )
    summary.update(receiving_landing_statistics(receiving))
    for i, row in enumerate(rows):
      row["robot_failed"] = bool(robot_failed[i])
      row["ball_return_success"] = bool(success[i])
      row["receiving_task_success"] = bool(success[i]) and not bool(robot_failed[i])
    labels: list[str] = [str(row["hit_type"]) for row in rows]
    summary["by_hit_type"] = {}
    for label in sorted(set(labels)):
      mask = torch.tensor(
        [value == label for value in labels], dtype=torch.bool, device=device
      )
      summary["by_hit_type"][label] = {
        "episodes": int(mask.sum()),
        "hit_rate": float(hit[mask].mean()),
        "net_clear_rate": float(receiving["ball/state_net_cleared"][mask].mean()),
        "receiving_task_success_rate": float(
          (success.bool()[mask] & ~robot_failed[mask]).float().mean()
        ),
        "robot_failure_rate": float(robot_failed[mask].float().mean()),
      }
  return {"summary": summary, "episodes": rows}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument(
    "--motion-directory",
    type=Path,
    help="Override the motion set; by default use the selected task configuration.",
  )
  parser.add_argument(
    "--tracker-file", type=Path, help="Frozen Stage-1 tracker for receiving tasks."
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--fixed-speed",
    action="store_true",
    help="Stage 2: set final high-level action to zero, fixing reference speed at 1.",
  )
  parser.add_argument(
    "--zero-residual",
    action="store_true",
    help="Stage 2: set all joint residual actions to zero, retaining the frozen tracker.",
  )
  parser.add_argument(
    "--arrival-time-jitter",
    type=float,
    help="Override seeded uniform arrival-time jitter half-width in seconds; 0 uses nominal timing.",
  )
  args = parser.parse_args()
  if args.repeats < 1:
    parser.error("--repeats must be positive.")
  if args.arrival_time_jitter is not None and args.arrival_time_jitter < 0:
    parser.error("--arrival-time-jitter must be nonnegative.")
  policies = [resolve_policy_artifacts(path) for path in args.checkpoint]
  configure_torch_backends(deterministic=True)
  cfg = load_env_cfg(args.task, play=True)
  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionCommandCfg):
    parser.error(f"Task {args.task!r} does not use a motion command.")
  if args.motion_directory is None:
    files = [Path(path).resolve() for path in _resolve_motion_paths(motion_cfg)]
  else:
    files = sorted(args.motion_directory.resolve().glob("*.npz"))
  if not files:
    parser.error(
      "No motions in the task configuration"
      if args.motion_directory is None
      else f"No motions in {args.motion_directory}"
    )
  cfg.seed = args.seed
  cfg.scene.num_envs = len(files) * args.repeats
  cfg.events = {}
  motion_cfg.motion_file = ""
  motion_cfg.motion_files = tuple(str(path) for path in files)
  motion_cfg.motion_directory = ""
  motion_cfg.sampling_mode = "start"
  motion_cfg.random_dt_training_enabled = False
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0, 0)
  for group in cfg.observations.values():
    group.enable_corruption = False
  for command in cfg.commands.values():
    command.debug_vis = False
  ball_cfg_term = cfg.commands.get("ball")
  if ball_cfg_term is not None and not isinstance(
    ball_cfg_term, ReceivingBallCommandCfg
  ):
    parser.error(f"Task {args.task!r} has an unsupported ball command.")
  ball_cfg = ball_cfg_term
  if ball_cfg is None and (
    args.fixed_speed or args.zero_residual or args.arrival_time_jitter is not None
  ):
    parser.error(
      "Receiving ablations and arrival jitter require a Stage-2 receiving task."
    )
  if ball_cfg is not None:
    for field in ("position_noise", "velocity_noise"):
      if hasattr(ball_cfg, field):
        setattr(ball_cfg, field, 0.0)
    if args.arrival_time_jitter is not None:
      ball_cfg.arrival_time_jitter = args.arrival_time_jitter
  joint_action_cfg = cfg.actions["joint_pos"]
  if args.tracker_file:
    if not isinstance(joint_action_cfg, FrozenTrackerActionCfg):
      parser.error("--tracker-file requires a Stage-2 frozen tracker action.")
    joint_action_cfg.tracker_file = str(resolve_jit(args.tracker_file))
  tracker_path = (
    Path(joint_action_cfg.tracker_file).expanduser().resolve()
    if isinstance(joint_action_cfg, FrozenTrackerActionCfg)
    else None
  )
  tracker_digest = sha256(tracker_path) if tracker_path is not None else None
  expected_tracker_digests = [
    validate_policy_tracker_dependency(policy, tracker_digest) for policy in policies
  ]
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  report = {
    "task": args.task,
    "seed": args.seed,
    "num_motions": len(files),
    "repeats": args.repeats,
    "protocol": "Every clip starts at frame zero, no observation/reset/domain noise, same seed per checkpoint; only the first episode per environment counts.",
    "timing": (
      "Reference speed fixed at 1 by the Stage-2 action ablation."
      if args.fixed_speed
      else "Fixed reference speed 1 for Stage 1; learned timing retained for Stage 2."
    ),
    "ablations": {"fixed_speed": args.fixed_speed, "zero_residual": args.zero_residual},
    "receiving_failure_definition": "Robot anchor/feet termination or ball.missed; valid ball_finished landings are not failures. Task success additionally requires a valid return and no robot failure."
    if ball_cfg is not None
    else None,
    "arrival_time_jitter_s": ball_cfg.arrival_time_jitter
    if ball_cfg is not None
    else None,
    "motions": [{"file": path.name, "sha256": sha256(path)} for path in files],
    "tracker": (
      {
        "path": str(tracker_path),
        "sha256": tracker_digest,
      }
      if tracker_path is not None
      else None
    ),
    "checkpoints": [],
  }
  try:
    for artifacts, expected_tracker_digest in zip(
      policies, expected_tracker_digests, strict=True
    ):
      policy = torch.jit.load(str(artifacts.jit), map_location=args.device).eval()
      evaluated = evaluate_policy(
        env,
        policy,
        files,
        args.seed,
        fixed_speed=args.fixed_speed,
        zero_residual=args.zero_residual,
      )
      evaluated.update(policy_provenance(artifacts, expected_tracker_digest))
      report["checkpoints"].append(evaluated)
      print(f"{artifacts.requested}: {json.dumps(evaluated['summary'])}")
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2) + "\n")
      del policy
  finally:
    env.close()
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()

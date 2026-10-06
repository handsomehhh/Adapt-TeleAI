"""Export early/late TensorBoard statistics and standalone A3 training curves.

Usage: uv run --no-sync python scripts/tools/analyze_a3_training.py LOG_DIRECTORY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize_series(events: list, window: int) -> dict:
  """Use iteration order, deduplicating resumed events at the same iteration."""
  values_by_step = {int(event.step): float(event.value) for event in events}
  ordered = sorted(values_by_step.items())
  steps = np.asarray([item[0] for item in ordered])
  values = np.asarray([item[1] for item in ordered])
  finite = np.isfinite(values)
  clean = values[finite]
  count = min(window, len(clean))
  return {
    "count": len(values),
    "finite": bool(finite.all()),
    "first_step": int(steps[0]),
    "last_step": int(steps[-1]),
    "window": count,
    "early_mean": float(clean[:count].mean()) if count else None,
    "late_mean": float(clean[-count:].mean()) if count else None,
    "last": float(values[-1]) if finite[-1] else None,
    "minimum": float(clean.min()) if count else None,
    "maximum": float(clean.max()) if count else None,
    "early_late_windows_overlap": len(clean) < 2 * window,
  }


def _configured_command_names(log_directory: Path) -> set[str] | None:
  """Read top-level command names from the serialized environment config."""
  path = log_directory / "params" / "env.yaml"
  if not path.is_file():
    return None
  names: set[str] = set()
  in_commands = False
  found_commands = False
  for line in path.read_text().splitlines():
    if line == "commands:":
      in_commands = True
      found_commands = True
      continue
    if not in_commands:
      continue
    if line and not line.startswith(" "):
      break
    if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
      names.add(line.strip().removesuffix(":"))
  return names if found_commands else None


def is_receiving_run(tags: list[str], log_directory: Path) -> bool:
  """Identify Stage 2 from metrics, saved config, or a robot-neutral run path."""
  if any(tag.startswith("Metrics/ball/") for tag in tags):
    return True
  command_names = _configured_command_names(log_directory)
  if command_names is not None:
    return "ball" in command_names
  agent_cfg = log_directory / "params" / "agent.yaml"
  if agent_cfg.is_file():
    for line in agent_cfg.read_text().splitlines():
      if line.startswith("experiment_name:"):
        return "pingpong_receive_stage2" in line.casefold()
  return "pingpong_receive_stage2" in str(log_directory).casefold()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("log_directory", type=Path)
  parser.add_argument("--output-directory", type=Path)
  parser.add_argument("--window", type=int, default=100)
  args = parser.parse_args()
  if args.window < 1:
    parser.error("--window must be positive.")
  out = args.output_directory or args.log_directory / "validation"
  out.mkdir(parents=True, exist_ok=True)
  accumulator = EventAccumulator(str(args.log_directory), size_guidance={"scalars": 0})
  accumulator.Reload()
  tags = accumulator.Tags()["scalars"]
  if not tags:
    parser.error(f"No scalar TensorBoard data in {args.log_directory}")
  series = {tag: accumulator.Scalars(tag) for tag in tags if accumulator.Scalars(tag)}
  summaries = {
    tag: summarize_series(events, args.window) for tag, events in series.items()
  }
  is_receiving = is_receiving_run(tags, args.log_directory)
  if is_receiving:
    episode_interpretation = (
      "Stage-2 episode length is in physical simulator control steps (0.02s per step "
      "in this task). Receiving episodes normally end when the actual ball outcome is "
      "known; they do not need to finish the reference motion. The ball_finished term "
      "includes valid return landings, which are normal task completion, not robot "
      "failure. Interpret length together with ball outcomes and anchor/feet failures, "
      "not as a full-motion completion score. Episode_Termination scalars are counts "
      "averaged across rollout control steps, not probabilities; terms can overlap. "
      "For acceptance use seeded evaluations that count only the first episode per "
      "environment, reporting receiving_task_success_rate, hit/net/return rates and "
      "robot_failure_rate. Report landing error conditioned on landing or successful "
      "return, with its sample count, to exclude zero defaults for shots never landed."
    )
  else:
    episode_interpretation = (
      "Episode length is in simulator control steps. Stage-1 training resets at adaptive "
      "reference phases and advances reference time by randomized dt; therefore a completed "
      "training episode is generally shorter than the full 94-frame clip. Reference clips "
      "span 1.86s; at fixed .02s reference/control dt a full clip lasts 93 control intervals, "
      "plus the .25s warmdown. Randomized reference dt is .01-.04s (mean .025s). "
      "Completion_fraction measures progress from the sampled start to clip end at reset, "
      "and is not the success rate of full clips from frame zero. Episode_Termination "
      "scalars are counts averaged across rollout control steps, not probabilities; "
      "failure terms can overlap. Use seeded frame-zero rollout completion for acceptance."
    )
  report = {
    "log_directory": str(args.log_directory.resolve()),
    "statistics_window_iterations": args.window,
    "all_logged_scalars_finite": all(item["finite"] for item in summaries.values()),
    "episode_length_interpretation": episode_interpretation,
    "metrics": summaries,
  }
  # Static report intended for export and review alongside saved checkpoints.
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  panels = [
    ("Return", ["Train/mean_reward"]),
    ("Episode length (control steps)", ["Train/mean_episode_length"]),
    (
      "Terminations (mean count per control step)",
      [tag for tag in tags if tag.startswith("Episode_Termination/")],
    ),
    (
      "Tracking errors",
      ["Metrics/motion/error_body_pos", "Metrics/motion/error_joint_pos"],
    ),
    ("Optimization", ["Loss/surrogate", "Loss/value"]),
    (
      "Reference timing / completion",
      ["Metrics/motion/reference_dt", "Metrics/motion/completion_fraction"],
    ),
  ]
  receive_metrics = [
    tag
    for tag in tags
    if any(
      word in tag.lower()
      for word in ("hit_rate", "contact_rate", "return_success", "landing_error")
    )
  ]
  if receive_metrics:
    panels[-1] = ("Receiving outcomes", receive_metrics)
  fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
  for axis, (title, wanted) in zip(axes.flat, panels, strict=True):
    for tag in wanted:
      if tag not in series:
        continue
      by_step = sorted(
        {int(event.step): float(event.value) for event in series[tag]}.items()
      )
      steps = np.asarray([item[0] for item in by_step])
      values = np.asarray([item[1] for item in by_step])
      smooth = min(20, max(1, len(values) // 20))
      axis.plot(steps, values, alpha=0.18, linewidth=0.6)
      if smooth > 1:
        axis.plot(
          steps[smooth - 1 :],
          np.convolve(values, np.ones(smooth) / smooth, mode="valid"),
          label=tag.split("/")[-1],
        )
      else:
        axis.plot(steps, values, label=tag.split("/")[-1])
    axis.set_title(title)
    axis.set_xlabel("PPO iteration")
    axis.grid(alpha=0.25)
    if axis.lines:
      axis.legend(fontsize=8)
  fig.suptitle(args.log_directory.name)
  fig.savefig(out / "training_curves.png", dpi=180)
  fig.savefig(out / "training_curves.pdf")
  plt.close(fig)
  (out / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"Wrote {out / 'training_summary.json'} and training_curves.png/.pdf")
  for tag in (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/motion_finished",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
  ):
    if tag in summaries:
      item = summaries[tag]
      print(f"{tag}: early={item['early_mean']:.6g}, late={item['late_mean']:.6g}")


if __name__ == "__main__":
  main()

"""Summarize host timing; optional torque alignment is not a causal delay test."""

import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values, *, scale=1000):
  values = np.asarray(values, dtype=float)
  finite = values[np.isfinite(values)]
  valid = finite[finite >= 0]
  return {
    "count": len(valid),
    "invalid_or_missing": len(values) - len(finite),
    "negative": int(np.sum(finite < 0)),
    "median": float(np.median(valid) * scale) if len(valid) else None,
    "p95": float(np.quantile(valid, 0.95) * scale) if len(valid) else None,
    "p99": float(np.quantile(valid, 0.99) * scale) if len(valid) else None,
    "max": float(np.max(valid) * scale) if len(valid) else None,
  }


def load_chunks(folder, pattern):
  files = sorted(Path(folder).glob(pattern))
  return (
    np.concatenate([np.load(p, allow_pickle=False) for p in files]) if files else None
  )


def torque_alignment(writes, states, names):
  """Compare tau_est with PD effort at candidate lags, retaining every lag."""
  curves = {}
  good = np.isfinite(writes["write_completed_time"])
  writes = writes[good]
  if not len(writes) or states is None or len(states) < 20:
    return curves
  times = writes["write_completed_time"]
  if np.any(np.diff(times) < 0):
    return {"error": "Write times are not monotonic"}
  for joint in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
    if joint not in names:
      continue
    i = names.index(joint)
    curve = []
    for milliseconds in range(41):
      indices = (
        np.searchsorted(
          times, states["callback_time"] - milliseconds / 1000, side="right"
        )
        - 1
      )
      valid = indices >= 0
      indices = np.maximum(indices, 0)
      valid &= writes["halted"][indices] == 0
      predicted = (
        writes["kp"][indices, i] * (writes["q_target"][indices, i] - states["q"][:, i])
        - writes["kd"][indices, i] * states["dq"][:, i]
      )
      observed = states["tau_est"][:, i]
      valid &= np.isfinite(predicted) & np.isfinite(observed)
      correlation = None
      if (
        valid.sum() >= 20
        and np.std(predicted[valid]) > 1e-5
        and np.std(observed[valid]) > 1e-5
      ):
        correlation = float(np.corrcoef(predicted[valid], observed[valid])[0, 1])
      curve.append(
        {
          "candidate_lag_ms": milliseconds,
          "correlation": correlation,
          "samples": int(valid.sum()),
        }
      )
    curves[joint] = curve
  return curves


def summarize(folder, *, response=False):
  folder = Path(folder)
  summary_file = folder / "summary.json"
  summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
  trace = folder / "trace.jsonl"
  rows = (
    [json.loads(line) for line in trace.read_text().splitlines()]
    if trace.exists()
    else []
  )
  writes = load_chunks(folder / "command_writes", "commands_*.npy")
  states = load_chunks(folder / "lowstate", "lowstate_*.npy")
  result = {
    "schema_version": 1,
    "units": "milliseconds unless explicitly labelled",
    "runtime_revision": summary.get("software", {}).get("runtime_revision"),
    "limitations": [
      "Host monotonic timestamps are comparable across local processes; source_tick is an unsynchronized device counter.",
      "DDS Write completion is not robot receipt or motor actuation acknowledgement.",
      "These measurements exclude sensing/filtering before host callback and transport/drive dynamics after Write.",
      "20ms policy period and 2ms writer period are update periods, not measured one-way communication delays.",
      "Closed-loop torque correlation can be aliased by oscillation, filtering and load; it cannot independently identify delay or prove the cause of oscillation.",
    ],
    "trace_frames": len(rows),
    "raw_feedback_frames": 0 if states is None else len(states),
    "actual_write_attempts": 0 if writes is None else len(writes),
    "phases": {},
    "recording_quality": summary.get("final_io_diagnostics", {}),
  }
  phase_by_cycle = {r.get("cycle_index"): r["state"] for r in rows}
  generated_ids = {
    r.get("latency", {}).get("command_id")
    for r in rows
    if r.get("command_generated_time") is not None
  }
  generated_ids.discard(None)
  if writes is not None:
    first = writes[(writes["first_write"] != 0) & (writes["halted"] == 0)]
    written_ids = set(first["command_id"].tolist())
    successful_ids = set(
      writes["command_id"][
        (writes["write_result"] == 1) & (writes["halted"] == 0)
      ].tolist()
    )
    result["command_correlation"] = {
      "generated_ids_in_trace": len(generated_ids),
      "ids_with_first_write": len(written_ids),
      "ids_with_Write_returning_true": len(successful_ids),
      "generated_without_observed_first_write": sorted(generated_ids - written_ids),
      "write_without_trace_id": sorted(written_ids - generated_ids),
      "generated_without_observed_Write_returning_true": sorted(
        generated_ids - successful_ids
      ),
      "note": "Missing writes may indicate replaced/dropped commands or lost records; inspect queue_put_succeeded and recorder dropped/error counts.",
    }
    for phase in dict.fromkeys(r["state"] for r in rows):
      selected = first[
        np.array(
          [phase_by_cycle.get(int(c)) == phase for c in first["control_cycle"]],
          dtype=bool,
        )
      ]
      rr = [r for r in rows if r["state"] == phase]
      metrics = {}
      for label, finish, start in (
        (
          "callback_to_parent_consume",
          "feedback_consumed_time",
          "feedback_callback_time",
        ),
        (
          "callback_to_feedback_packet_enqueue",
          "feedback_packet_enqueued_time",
          "feedback_callback_time",
        ),
        (
          "feedback_ipc_queue",
          "feedback_packet_dequeued_time",
          "feedback_packet_enqueued_time",
        ),
        (
          "parent_consume_to_command_generation",
          "generated_time",
          "feedback_consumed_time",
        ),
        ("command_generation_to_acceptance", "accepted_time", "generated_time"),
        ("command_generation_to_io_dequeue", "command_dequeued_time", "generated_time"),
        ("io_dequeue_to_acceptance", "accepted_time", "command_dequeued_time"),
        ("command_generation_to_first_write", "write_started_time", "generated_time"),
        ("writer_preparation", "write_started_time", "prepare_started_time"),
        ("DDS_Write_call", "write_completed_time", "write_started_time"),
        ("callback_to_first_write", "write_started_time", "feedback_callback_time"),
      ):
        metrics[label] = distribution(
          selected[finish] - selected[start]
          if finish in selected.dtype.names and start in selected.dtype.names
          else np.full(len(selected), np.nan)
        )
      metrics["policy_loop_period"] = distribution(
        [r.get("loop_period_seconds", np.nan) or np.nan for r in rr]
      )
      metrics["reused_feedback_frames"] = sum(
        bool(r.get("latency", {}).get("feedback_reused")) for r in rr
      )
      metrics["trace_commands_not_queued"] = sum(
        r.get("latency", {}).get("command_queue_put_succeeded") is False for r in rr
      )
      result["phases"][phase] = metrics
    result["all_write_intervals"] = distribution(writes["write_interval_seconds"])
    result["all_DDS_Write_call_durations"] = distribution(
      writes["write_completed_time"] - writes["write_started_time"]
    )
    result["all_write_schedule_lateness"] = distribution(
      writes["prepare_started_time"] - writes["scheduled_time"]
    )
    result["damping_write_attempts"] = int(np.sum(writes["halted"] != 0))
    result["writes_returning_false"] = int(np.sum(writes["write_result"] == 0))
    if response:
      schema = json.loads((folder / "command_writes/schema.json").read_text())
      result["exploratory_torque_alignment"] = torque_alignment(
        writes, states, schema["joint_names"]
      )
  else:
    result["command_write_journal_missing"] = True
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recording", type=Path)
  parser.add_argument(
    "--response",
    action="store_true",
    help="Include exploratory waist torque correlation curves, not a causal delay estimate",
  )
  args = parser.parse_args()
  result = summarize(args.recording, response=args.response)
  destination = args.recording / "latency_summary.json"
  destination.write_text(json.dumps(result, indent=2))
  print(destination)


if __name__ == "__main__":
  main()

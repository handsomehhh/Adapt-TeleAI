"""Offline, measured-state analysis of commands, x-hit crossings and racket gaps."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def interpolate(times, values, t, max_gap=0.05):
  i = int(np.searchsorted(times, t))
  if i < len(times) and abs(times[i] - t) < 1e-9:
    return np.asarray(values[i])
  if i == 0 or i == len(times) or not 0 < times[i] - times[i - 1] <= max_gap:
    return None
  f = (t - times[i - 1]) / (times[i] - times[i - 1])
  return np.asarray(values[i - 1]) * (1 - f) + np.asarray(values[i]) * f


def crossings(times, points, x, start, end):
  result = []
  for i in range(
    max(1, np.searchsorted(times, start)),
    min(len(times), np.searchsorted(times, end, side="right")),
  ):
    a, b = points[i - 1 : i + 1]
    dt = times[i] - times[i - 1]
    if 0 < dt <= 0.05 and a[0] > x >= b[0]:
      f = (a[0] - x) / (a[0] - b[0])
      result.append((times[i - 1] + f * dt, a + f * (b - a)))
  return result


def load(path):
  return json.loads(path.read_text())


def rows(path):
  with path.open() as stream:
    for line in stream:
      yield json.loads(line)


def command_error(command, deadline, start, bt, bp):
  goal = np.asarray(command["position"])
  found = crossings(bt, bp, goal[0], start - 0.02, deadline + 0.5)
  result = dict(
    command_position=goal.tolist(),
    deadline=deadline,
    crossing_candidates=len(found),
    crossing_position=None,
    planner_position_error_m=None,
    planner_time_error_ms=None,
  )
  if found:
    stamp, actual = min(found, key=lambda item: abs(item[0] - deadline))
    result.update(
      crossing_time=stamp,
      crossing_position=actual.tolist(),
      planner_delta_m=(goal - actual).tolist(),
      planner_position_error_m=float(np.linalg.norm(goal - actual)),
      planner_time_error_ms=float((deadline - stamp) * 1000),
    )
  return result


def analyze(folder):
  summary = load(folder / "summary.json")
  trace = list(rows(folder / "trace.jsonl"))
  origin = trace[0]["time"]
  rt = np.array([r["time"] - origin for r in trace])
  usable = [r for r in trace if r.get("racket_pos") is not None]
  kt = np.array([r["time"] - origin for r in usable])
  kp = np.array([r["racket_pos"] for r in usable])
  kn = np.array([r["racket_normal"] for r in usable])
  packets = list(rows(folder / "mocap_packets.jsonl"))
  ball = sorted(
    (r for r in packets if r["packet"]["kind"] == "ball"),
    key=lambda r: r["arrival_time"],
  )
  bt = np.array([r["arrival_time"] - origin for r in ball])
  bp = np.array([r["packet"]["position"] for r in ball])
  received = []
  for record in packets:
    c = record["packet"]
    if c["kind"] != "racket_command" or not c.get("valid"):
      continue
    start = record["arrival_time"] - origin
    deadline = (
      c.get("bridge_monotonic", record["arrival_time"]) - origin + c["time_to_strike"]
    )
    received.append(
      dict(received_time=start, **command_error(c, deadline, start, bt, bp))
    )
  swings = []
  events = [e for e in summary["events"] if e["event"] == "motion_selected"]
  for number, event in enumerate(events, 1):
    c = event["raw_command"]
    start, deadline = event["time"] - origin, event["deadline"] - origin
    result = dict(
      swing=number,
      start=start,
      motion=event["motion_index"],
      tts_ms=(deadline - start) * 1000,
      **command_error(c, deadline, start, bt, bp),
    )
    goal = np.array(c["position"])
    cv, cn = np.array(c["velocity"]), np.array(c["normal"])

    def state_at(t):
      p = interpolate(kt, kp, t, 0.06)
      n = interpolate(kt, kn, t, 0.06)
      b = interpolate(bt, bp, t)
      return p, n / np.linalg.norm(n) if n is not None else None, b

    p, n, b = state_at(deadline)
    if p is not None:
      result.update(
        racket_at_deadline=p.tolist(),
        racket_command_delta_m=(p - goal).tolist(),
        racket_command_error_m=float(np.linalg.norm(p - goal)),
      )
      ids = np.flatnonzero(np.abs(kt - deadline) <= 0.03)
      if len(ids) >= 2:
        velocity = np.linalg.lstsq(
          np.c_[kt[ids] - deadline, np.ones(len(ids))], kp[ids], rcond=None
        )[0][0]
        result.update(
          racket_velocity=velocity.tolist(),
          command_velocity=cv.tolist(),
          velocity_error_mps=float(np.linalg.norm(velocity - cv)),
          racket_speed_mps=float(np.linalg.norm(velocity)),
          command_speed_mps=float(np.linalg.norm(cv)),
        )
      result["normal_error_degrees"] = float(
        np.degrees(np.arccos(np.clip(abs(n @ cn / np.linalg.norm(cn)), 0, 1)))
      )
      if b is not None:
        result.update(
          ball_at_deadline=b.tolist(),
          racket_ball_at_deadline_m=float(np.linalg.norm(p - b)),
          ball_command_at_deadline_m=float(np.linalg.norm(b - goal)),
        )
    crossing = result.get("crossing_time")
    if crossing is not None:
      p, n, b = state_at(crossing)
      if p is not None:
        result["racket_ball_at_crossing_m"] = float(
          np.linalg.norm(p - result["crossing_position"])
        )
    closest, disk_closest = None, None
    for t in np.arange(max(start, deadline - 0.25), deadline + 0.25, 0.001):
      p, n, b = state_at(t)
      if p is None or b is None:
        continue
      delta = b - p
      axial = abs(float(delta @ n))
      radial = float(np.sqrt(max(0, delta @ delta - axial**2)))
      radius = summary["rally_runtime_config"]["ball"].get("racket_radius", 0.075)
      disk_distance = float(np.hypot(axial, max(0, radial - radius)))
      sample = dict(
        time=float(t),
        dt_ms=float((t - deadline) * 1000),
        distance_m=float(np.linalg.norm(delta)),
        axial_m=axial,
        radial_m=radial,
        disk_distance_m=disk_distance,
        ball_position=b.tolist(),
        racket_position=p.tolist(),
      )
      if closest is None or sample["distance_m"] < closest["distance_m"]:
        closest = sample
      if disk_closest is None or disk_distance < disk_closest["disk_distance_m"]:
        disk_closest = sample
    result["closest_centers"] = closest
    result["closest_disk"] = disk_closest
    # Observed X reversal is evidence of a deflection, not proof of paddle contact.
    window = np.flatnonzero((bt > deadline - 0.12) & (bt < deadline + 0.20))
    if len(window):
      index = window[np.argmin(bp[window, 0])]
      turn = bt[index]

      def fitted_velocity(lo, hi, turn=turn):
        ids = np.flatnonzero((bt >= turn + lo) & (bt <= turn + hi))
        if len(ids) < 3 or np.max(np.diff(bt[ids])) > 0.05:
          return None
        return np.linalg.lstsq(
          np.c_[bt[ids] - turn, np.ones(len(ids))], bp[ids], rcond=None
        )[0][0]

      incoming = fitted_velocity(-0.06, -0.02)
      outgoing = fitted_velocity(0.02, 0.07)
      if (
        incoming is not None
        and outgoing is not None
        and incoming[0] < -0.5
        and outgoing[0] > 0.5
      ):
        near_disk = disk_closest is not None and abs(turn - disk_closest["time"]) < 0.03
        result["observed_deflection"] = dict(
          time=float(turn),
          dt_ms=float((turn - deadline) * 1000),
          position=bp[index].tolist(),
          incoming_velocity=incoming.tolist(),
          outgoing_velocity=outgoing.tolist(),
          planner_outgoing_velocity=c.get("outgoing_velocity"),
          near_racket_time=bool(near_disk),
          interpretation="near racket encounter"
          if near_disk
          else "later deflection; body/contact identity unknown",
        )
        pturn, nturn, _ = state_at(turn)
        if nturn is not None:
          result["observed_deflection"]["normal_error_degrees"] = float(
            np.degrees(np.arccos(np.clip(abs(nturn @ cn / np.linalg.norm(cn)), 0, 1)))
          )
        # First observed outgoing net crossing and descending table-height crossing.
        net = summary["rally_runtime_config"]["ball"].get("net_x", 1.37)
        for j in range(index + 1, min(len(bt), np.searchsorted(bt, turn + 1.5))):
          a, b = bp[j - 1 : j + 1]
          if not 0 < bt[j] - bt[j - 1] <= 0.05:
            result["outgoing_tracking_gap"] = True
            break
          if a[0] < net <= b[0] and "observed_net_crossing" not in result:
            f = (net - a[0]) / (b[0] - a[0])
            result["observed_net_crossing"] = (a + f * (b - a)).tolist()
          if a[2] > 0.02 >= b[2]:
            f = (0.02 - a[2]) / (b[2] - a[2])
            point = a + f * (b - a)
            result["first_descending_table_height"] = point.tolist()
            result["observed_opponent_landing_geometry"] = bool(
              net <= point[0] <= 2.74 and -1.525 <= point[1] <= 0
            )
            break
    near = np.flatnonzero(np.abs(kt - deadline) <= 0.25)
    if len(near):
      index = near[np.argmin(np.linalg.norm(kp[near] - goal, axis=1))]
      result.update(
        racket_closest_command_error_m=float(np.linalg.norm(kp[index] - goal)),
        racket_closest_command_time_ms=float((kt[index] - deadline) * 1000),
      )
    swings.append(result)
  x = summary["rally_runtime_config"].get("a3", {}).get("hit_plane_x", 0)
  incoming_crossings = crossings(bt, bp, x, 0, rt[-1])
  return dict(
    recording=folder.name,
    duration_s=float(rt[-1]),
    origin=origin,
    frames=len(trace),
    ball_samples=len(ball),
    received_valid_commands=len(received),
    completed_swings=summary.get("completed_swings"),
    recovered_swings=summary.get("recovered_swings"),
    incoming_plane_crossings=len(incoming_crossings),
    plane_x=x,
    decisions=dict(
      Counter((r.get("decision") or {}).get("reason", "none") for r in trace)
    ),
    states=dict(Counter(r["state"] for r in trace)),
    source_ball_clock="local arrival; capture timestamps not independently available",
    ball_radius=summary["rally_runtime_config"]["ball"].get("ball_radius", 0.02),
    swings=swings,
    received_commands=received,
  )


def write_csv(path, records):
  # Keep nested vectors/diagnostics as JSON cells.
  if not records:
    path.write_text("")
    return
  fields = list(dict.fromkeys(k for r in records for k in r))
  with path.open("w") as stream:
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(
      {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
      for row in records
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recordings", type=Path, nargs="+")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  reports = []
  for folder in args.recordings:
    report = analyze(folder)
    reports.append(report)
    (args.output / (folder.name + ".json")).write_text(json.dumps(report, indent=2))
    write_csv(args.output / (folder.name + "_swings.csv"), report["swings"])
    write_csv(
      args.output / (folder.name + "_commands.csv"), report["received_commands"]
    )
    print(
      folder.name,
      "swings",
      len(report["swings"]),
      "commands",
      len(report["received_commands"]),
    )
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
  for report in reports:
    swings = report["swings"]
    if not swings:
      continue
    ids = [s["swing"] for s in swings]
    for key, label in [
      ("planner_position_error_m", "command vs measured crossing"),
      ("racket_command_error_m", "racket vs command @ deadline"),
      ("racket_ball_at_deadline_m", "racket vs ball @ deadline"),
    ]:
      axes[0].plot(
        ids,
        [
          s.get(key, np.nan) * 100 if s.get(key) is not None else np.nan for s in swings
        ],
        "o-",
        label=label,
      )
    axes[1].plot(
      ids,
      [s.get("planner_time_error_ms", np.nan) for s in swings],
      "o-",
      label="deadline - measured crossing",
    )
    axes[1].plot(
      ids,
      [s.get("racket_closest_command_time_ms", np.nan) for s in swings],
      "o-",
      label="racket closest target time - deadline",
    )
    axes[3].plot(
      ids,
      [s.get("normal_error_degrees", np.nan) for s in swings],
      "o-",
      label="normal error (deg)",
    )
    axes[2].plot(
      ids,
      [s.get("velocity_error_mps", np.nan) for s in swings],
      "o-",
      label="velocity vector error (m/s)",
    )
  for ax, label in zip(
    axes,
    [
      "position error (cm)",
      "timing error (ms)",
      "velocity error (m/s)",
      "normal error (deg)",
    ],
    strict=True,
  ):
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend()
  axes[-1].set_xlabel("executed swing index")
  fig.tight_layout()
  fig.savefig(args.output / "errors.png", dpi=160)


if __name__ == "__main__":
  main()

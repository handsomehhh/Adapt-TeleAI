"""Measure whether table-restitution mismatch explains next-x_hit error.

For every valid table impact this tool estimates an incident and outgoing
velocity from fixed windows around the measured contact.  It then starts the
normal Stage-2 model *at the measured pre-impact state*, so its first action is
the configured table bounce.  Only flights whose recorded trajectory naturally
crosses x_hit before the next table bounce are retained: trajectories changed
by the robot racket are deliberately excluded.

For each retained flight we also run a counterfactual prediction whose C_h/C_v
are the effective ratios measured at that same impact.  The gap between the
two predictions is the portion of the x_hit error attributable to restitution
in this model (rather than state-estimation error before the bounce).
"""

import argparse
import csv
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path

import numpy as np

from .ball_trajectory_predictor import BallTrajectoryPredictor
from .benchmark_planner_timing import load_planner_settings
from .constants import TableParams


BALL_COLUMNS = ("ball_stamp_s", "ball_x", "ball_y", "ball_z")


def _load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(BALL_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(BALL_COLUMNS)}")
        for row in reader:
            try:
                value = [float(row[key]) for key in BALL_COLUMNS]
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(value)):
                rows.append(value)
    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return values[:, 0], values[:, 1:]


def _fit_velocity(times: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Least-squares linear velocity over a local, fixed observation window."""
    centered = times - times.mean()
    design = np.column_stack((centered, np.ones(len(centered))))
    return np.linalg.lstsq(design, points, rcond=None)[0][0]


def _find_impacts(times: np.ndarray, points: np.ndarray, radius: float):
    """Return isolated local z minima that have downward/upward velocity."""
    candidates = []
    for index in range(30, len(points) - 30):
        p = points[index]
        if not (-radius <= p[0] <= 2.74 + radius and -1.525 - radius <= p[1] <= radius):
            continue
        if p[2] >= 0.04 or p[2] != points[index - 4:index + 5, 2].min():
            continue
        vin = _fit_velocity(times[index - 25:index - 5], points[index - 25:index - 5])
        vout = _fit_velocity(times[index + 5:index + 25], points[index + 5:index + 25])
        if vin[0] < -0.5 and vin[2] < -1.0 and vout[2] > 1.0:
            if not candidates or index - candidates[-1] > 100:
                candidates.append(index)
    return candidates


def _first_natural_xhit(times, points, start, stop, x_hit):
    """Interpolate first incoming x_hit crossing before a later impact."""
    for index in range(max(1, start), stop):
        left, right = points[index - 1], points[index]
        if left[0] > x_hit >= right[0]:
            dt = times[index] - times[index - 1]
            if dt <= 0 or (right[0] - left[0]) / dt >= 0:
                continue
            frac = (left[0] - x_hit) / (left[0] - right[0])
            p = left + frac * (right - left)
            p[0] = x_hit
            return times[index - 1] + frac * dt, p
    return None


def _predict(p0, vin, t0, physics, config, table):
    return BallTrajectoryPredictor(physics, config, table).predict(p0, vin, t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ball_points_csv", type=Path)
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-flight-time", type=float, default=1.0,
                        help="reject a later x_hit caused by a new rally event (default: 1 s)")
    args = parser.parse_args()

    config, nominal = load_planner_settings(args.params_file)
    times, points = _load_samples(args.ball_points_csv)
    table = TableParams()
    impacts = _find_impacts(times, points, nominal.radius)
    fields = [
        "impact", "impact_frame", "impact_time_s", "effective_c_h", "effective_c_v",
        "delta_c_h", "delta_c_v", "coefficient_deviation",
        "actual_xhit_time_s", "actual_y_m", "actual_z_m",
        "nominal_valid", "nominal_y_m", "nominal_z_m", "nominal_time_s",
        "nominal_yz_error_m", "nominal_time_error_s",
        "measured_c_valid", "measured_c_y_m", "measured_c_z_m", "measured_c_time_s",
        "measured_c_yz_error_m", "measured_c_time_error_s",
        "restitution_explained_yz_m",
        "post_state_valid", "post_state_y_m", "post_state_z_m", "post_state_time_s",
        "post_state_yz_error_m", "post_state_time_error_s",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, index in enumerate(impacts, start=1):
            next_impact = impacts[number] if number < len(impacts) else len(points)
            actual = _first_natural_xhit(times, points, index + 1, next_impact, config.x_hit)
            if actual is None or actual[0] - times[index] > args.max_flight_time:
                continue

            vin = _fit_velocity(times[index - 25:index - 5], points[index - 25:index - 5])
            vout = _fit_velocity(times[index + 5:index + 25], points[index + 5:index + 25])
            c_h = float(np.linalg.norm(vout[:2]) / np.linalg.norm(vin[:2]))
            c_v = float(-vout[2] / vin[2])
            # Use the observed xy contact position but model's ball-centre table height.
            p0 = np.array([points[index, 0], points[index, 1], nominal.radius])
            nominal_strike = _predict(p0, vin, times[index], nominal, config, table)
            measured = replace(nominal, C_h=c_h, C_v=c_v)
            measured_strike = _predict(p0, vin, times[index], measured, config, table)
            # This skips the table model entirely: it is the residual error after
            # granting the predictor the measured outgoing state.
            post_state_strike = _predict(p0, vout, times[index], nominal, config, table)
            actual_t, actual_p = actual
            nominal_yz = float(np.linalg.norm(nominal_strike.p_ball[1:] - actual_p[1:]))
            measured_yz = float(np.linalg.norm(measured_strike.p_ball[1:] - actual_p[1:]))
            post_state_yz = float(np.linalg.norm(post_state_strike.p_ball[1:] - actual_p[1:]))
            writer.writerow({
                "impact": number, "impact_frame": index, "impact_time_s": f"{times[index]:.9f}",
                "effective_c_h": c_h, "effective_c_v": c_v,
                "delta_c_h": c_h - nominal.C_h, "delta_c_v": c_v - nominal.C_v,
                "coefficient_deviation": float(np.hypot(c_h - nominal.C_h, c_v - nominal.C_v)),
                "actual_xhit_time_s": f"{actual_t:.9f}", "actual_y_m": actual_p[1], "actual_z_m": actual_p[2],
                "nominal_valid": nominal_strike.valid, "nominal_y_m": nominal_strike.p_ball[1],
                "nominal_z_m": nominal_strike.p_ball[2], "nominal_time_s": nominal_strike.t_strike,
                "nominal_yz_error_m": nominal_yz, "nominal_time_error_s": nominal_strike.t_strike - actual_t,
                "measured_c_valid": measured_strike.valid, "measured_c_y_m": measured_strike.p_ball[1],
                "measured_c_z_m": measured_strike.p_ball[2], "measured_c_time_s": measured_strike.t_strike,
                "measured_c_yz_error_m": measured_yz, "measured_c_time_error_s": measured_strike.t_strike - actual_t,
                "restitution_explained_yz_m": nominal_yz - measured_yz,
                "post_state_valid": post_state_strike.valid,
                "post_state_y_m": post_state_strike.p_ball[1], "post_state_z_m": post_state_strike.p_ball[2],
                "post_state_time_s": post_state_strike.t_strike,
                "post_state_yz_error_m": post_state_yz,
                "post_state_time_error_s": post_state_strike.t_strike - actual_t,
            })
            kept += 1
    print(f"detected impacts: {len(impacts)}; natural x_hit flights retained: {kept}")
    print(f"nominal C_h={nominal.C_h:.4f}, C_v={nominal.C_v:.4f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

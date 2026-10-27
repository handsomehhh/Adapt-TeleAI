"""Compare every original Stage-1/Stage-2 rollout point with the actual ball path.

Each output row is one predicted point in one rollout.  The rollout starts from
the original quadratic-fit Stage 1 state and uses the package's normal Stage 2
trajectory predictor, so it is directly comparable with the EKF error CSV.
"""

import argparse
import csv
from bisect import bisect_left
from pathlib import Path

import numpy as np

from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor
from .benchmark_planner_timing import load_planner_settings
from .constants import TableParams


BALL_COLUMNS = ("ball_stamp_s", "ball_x", "ball_y", "ball_z")


def _load_samples(path: Path) -> list[tuple[float, np.ndarray]]:
    samples: list[tuple[float, np.ndarray]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(BALL_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain: {', '.join(BALL_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                t, x, y, z = (float(row[key]) for key in BALL_COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid ball sample") from exc
            if np.all(np.isfinite((t, x, y, z))):
                samples.append((t, np.array([x, y, z], dtype=float)))
    if len(samples) < 2:
        raise ValueError(f"{path}: need at least two finite ball samples")
    return samples


def _incoming_crossings(samples: list[tuple[float, np.ndarray]], x_hit: float):
    crossings = []
    for index in range(1, len(samples)):
        previous_t, previous = samples[index - 1]
        current_t, current = samples[index]
        if previous[0] > x_hit >= current[0] and current_t > previous_t:
            if (current[0] - previous[0]) / (current_t - previous_t) < 0.0:
                fraction = (previous[0] - x_hit) / (previous[0] - current[0])
                crossing = previous + fraction * (current - previous)
                crossing[0] = x_hit
                crossings.append((index, previous_t + fraction * (current_t - previous_t), crossing))
    return crossings


def _interpolate_actual(samples, times, query_time):
    index = bisect_left(times, query_time)
    if index == 0 or index == len(samples):
        return None
    left_t, left_p = samples[index - 1]
    right_t, right_p = samples[index]
    if right_t <= left_t:
        return None
    return left_p + ((query_time - left_t) / (right_t - left_t)) * (right_p - left_p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ball_points_csv", type=Path)
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument("--crossing-index", type=int, default=0)
    parser.add_argument("--start-seconds-before-crossing", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config, physics = load_planner_settings(args.params_file)
    samples = _load_samples(args.ball_points_csv)
    times = [time for time, _ in samples]
    crossings = _incoming_crossings(samples, config.x_hit)
    if not 0 <= args.crossing_index < len(crossings):
        raise ValueError(f"crossing-index must be in [0, {len(crossings) - 1}]")
    crossing_frame, crossing_time, _ = crossings[args.crossing_index]
    start_time = crossing_time - args.start_seconds_before_crossing
    start_frame = next(index for index, (time, _) in enumerate(samples) if time >= start_time)

    estimator = BallStateEstimator(config)
    predictor = BallTrajectoryPredictor(physics, config, TableParams())
    fields = [
        "origin_frame_index", "origin_time_s", "rollout_time_s", "horizon_s",
        "predicted_x_m", "predicted_y_m", "predicted_z_m",
        "actual_x_m", "actual_y_m", "actual_z_m",
        "x_error_m", "y_error_m", "z_error_m", "yz_error_m", "xyz_error_m",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    origins_written = rows_written = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for origin_index in range(start_frame, crossing_frame):
            origin_time, measurement = samples[origin_index]
            estimator.push(origin_time, measurement)
            if not estimator.ready:
                continue
            estimate, velocity, estimate_time = estimator.estimate()
            strike, rollout = predictor.predict_with_samples(estimate, velocity, estimate_time)
            if not strike.valid:
                continue
            origins_written += 1
            for rollout_time, predicted_x, predicted_y, predicted_z in rollout:
                if rollout_time > crossing_time + 1e-9:
                    break
                actual = _interpolate_actual(samples, times, float(rollout_time))
                if actual is None:
                    continue
                predicted = np.array([predicted_x, predicted_y, predicted_z])
                error = predicted - actual
                writer.writerow({
                    "origin_frame_index": origin_index,
                    "origin_time_s": f"{origin_time:.9f}",
                    "rollout_time_s": f"{rollout_time:.9f}",
                    "horizon_s": f"{rollout_time - origin_time:.9f}",
                    "predicted_x_m": predicted_x, "predicted_y_m": predicted_y, "predicted_z_m": predicted_z,
                    "actual_x_m": actual[0], "actual_y_m": actual[1], "actual_z_m": actual[2],
                    "x_error_m": error[0], "y_error_m": error[1], "z_error_m": error[2],
                    "yz_error_m": float(np.linalg.norm(error[1:])),
                    "xyz_error_m": float(np.linalg.norm(error)),
                })
                rows_written += 1
    print(f"analyzed original-estimator origins: {origins_written}; rollout point comparisons: {rows_written}")
    print(f"actual flight endpoint: x_hit at t={crossing_time:.9f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

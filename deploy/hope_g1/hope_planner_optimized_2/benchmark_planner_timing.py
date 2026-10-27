"""Offline timing benchmark for the HOPE planner's per-frame calculation.

The input is an E2E ``ball_points.csv`` trace.  It contains the original ball
timestamps and positions, so replaying it into :class:`HOPEPlanner` exercises
the normal estimator -> predictor -> racket-target pipeline without adding ROS
transport, publisher, CSV-recorder, or controller latency to the measurement.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from .constants import BallPhysics, PlannerConfig
from .planner import HOPEPlanner


PACKAGE_NAME = "hope_planner_optimized_2"
NODE_NAME = "hope_planner_optimized_2"
BALL_COLUMNS = ("ball_stamp_s", "ball_x", "ball_y", "ball_z")


def _default_params_path() -> Path:
    """Find the installed package's normal node parameter file."""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory(PACKAGE_NAME)) / "config" / "hope_planner.yaml"


def _float_param(params: dict[str, object], name: str, default: float) -> float:
    return float(params.get(name, default))


def load_planner_settings(params_path: Path) -> tuple[PlannerConfig, BallPhysics]:
    """Build exactly the planner configuration constructed by ``HOPEPlannerNode``."""
    with params_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    try:
        params = document[NODE_NAME]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{params_path} must contain {NODE_NAME}.ros__parameters"
        ) from exc
    if not isinstance(params, dict):
        raise ValueError(f"{params_path}: ros__parameters must be a mapping")

    config = PlannerConfig(
        x_hit=_float_param(params, "x_hit", 0.0),
        target_land=np.array([
            _float_param(params, "target_land_x", 2.055),
            _float_param(params, "target_land_y", -0.7625),
            0.0,
        ]),
        delta_t_flight=_float_param(params, "delta_t_flight", 0.5),
        C_r=_float_param(params, "restitution_racket", 0.88),
        # Stage 2 incoming trajectory-prediction integration step.
        dt_integrate=_float_param(params, "dt_integrate", 0.001),
        stage3_dt_integrate=_float_param(params, "stage3_dt_integrate", 0.001),
        outgoing_velocity_tolerance_m=_float_param(
            params, "outgoing_velocity_tolerance_m", 1e-4
        ),
        time_to_strike_invalid_x=_float_param(
            params, "time_to_strike_invalid_x", 0.0
        ),
        publish_intercept_x_min=_float_param(params, "publish_intercept_x_min", 0.0),
        publish_intercept_x_max=_float_param(params, "publish_intercept_x_max", 0.0),
        publish_intercept_y_min=_float_param(params, "publish_intercept_y_min", 0.0),
        publish_intercept_y_max=_float_param(params, "publish_intercept_y_max", 1.5),
        publish_intercept_z_min=_float_param(params, "publish_intercept_z_min", 0.2),
        publish_intercept_z_max=_float_param(params, "publish_intercept_z_max", 0.6),
    )
    physics = BallPhysics(
        k=_float_param(params, "drag_k", 0.5),
        C_h=_float_param(params, "restitution_h", 0.75),
        C_v=_float_param(params, "restitution_v", 0.85),
    )
    return config, physics


def resolve_ball_trace(path: str) -> Path:
    """Accept either an E2E trace directory or its ``ball_points.csv`` path."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "ball_points.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"ball trace not found: {candidate}")
    return candidate


def load_ball_samples(path: Path) -> tuple[list[tuple[float, np.ndarray]], int]:
    """Load finite ``ball_stamp_s, ball_x, ball_y, ball_z`` records from CSV."""
    samples: list[tuple[float, np.ndarray]] = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(BALL_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must have columns: {', '.join(BALL_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                t, x, y, z = (float(row[column]) for column in BALL_COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid ball sample") from exc
            if not all(math.isfinite(value) for value in (t, x, y, z)):
                skipped += 1
                continue
            samples.append((t, np.array([x, y, z], dtype=float)))
    if not samples:
        raise ValueError(f"{path}: no finite ball samples")
    return samples, skipped


def _summary_ms(durations_ns: Iterable[int]) -> dict[str, float]:
    values_ms = np.asarray(list(durations_ns), dtype=float) / 1e6
    if values_ms.size == 0:
        return {"count": 0.0}
    return {
        "count": float(values_ms.size),
        "mean": float(np.mean(values_ms)),
        "p50": float(np.percentile(values_ms, 50)),
        "p95": float(np.percentile(values_ms, 95)),
        "p99": float(np.percentile(values_ms, 99)),
        "max": float(np.max(values_ms)),
    }


def _format_summary(label: str, summary: dict[str, float]) -> str:
    if summary["count"] == 0:
        return f"  {label}: no frames"
    return (
        f"  {label}: n={int(summary['count'])}, mean={summary['mean']:.3f} ms, "
        f"p50={summary['p50']:.3f} ms, p95={summary['p95']:.3f} ms, "
        f"p99={summary['p99']:.3f} ms, max={summary['max']:.3f} ms"
    )


def benchmark_once(
    samples: list[tuple[float, np.ndarray]],
    config: PlannerConfig,
    physics: BallPhysics,
    collect_prediction_samples: bool,
    outgoing_velocity_tolerance_m: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], int]:
    """Replay one trace and return all-frame/full-planning timing summaries."""
    planner = HOPEPlanner(config=config, physics=physics)
    planner.target_planner.set_outgoing_velocity_tolerance_m(
        outgoing_velocity_tolerance_m
    )
    frame_durations_ns: list[int] = []
    planning_durations_ns: list[int] = []
    stage_durations_ns = {"stage1": [], "stage2": [], "stage3": [], "other": []}
    published_commands = 0

    for timestamp, position in samples:
        start_ns = time.perf_counter_ns()
        command = planner.update(
            timestamp,
            position,
            collect_prediction_samples=collect_prediction_samples,
            profile_stages=True,
        )
        duration_ns = time.perf_counter_ns() - start_ns
        frame_durations_ns.append(duration_ns)
        # _latest_strike is set only after Stage 2 has run.  This separates
        # inexpensive buffer/gate frames from the complete 1 -> 2 -> 3 path.
        if planner._latest_strike is not None:
            planning_durations_ns.append(duration_ns)
            stage_timings = planner.last_stage_timings_ns
            stage_total_ns = sum(stage_timings.values())
            for stage in ("stage1", "stage2", "stage3"):
                stage_durations_ns[stage].append(stage_timings[stage])
            stage_durations_ns["other"].append(max(0, duration_ns - stage_total_ns))
        if command is not None:
            published_commands += 1

    return (
        _summary_ms(frame_durations_ns),
        _summary_ms(planning_durations_ns),
        {stage: _summary_ms(durations) for stage, durations in stage_durations_ns.items()},
        published_commands,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an E2E ball_points.csv trace and benchmark HOPEPlanner.update() "
            "without ROS transport or recorder I/O."
        )
    )
    parser.add_argument(
        "trace",
        help="E2E trace directory containing ball_points.csv, or that CSV file directly",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="planner ROS parameter YAML (default: installed package config/hope_planner.yaml)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="measured replays after warm-up (default: 5)",
    )
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=1,
        help="unreported warm-up replays (default: 1)",
    )
    parser.add_argument(
        "--collect-prediction-samples",
        action="store_true",
        help="benchmark prediction-path allocation used by recording, but not CSV I/O",
    )
    parser.add_argument(
        "--outgoing-velocity-tolerance-mm",
        type=float,
        default=None,
        help=(
            "Newton/fallback target landing-position tolerance in mm; benchmark-only "
            "override (default: value from the parameter YAML)"
        ),
    )
    parser.add_argument(
        "--integration-dt-ms",
        type=float,
        default=None,
        help=(
            "Stage 2/3 integration step in ms; benchmark-only override "
            "(default: value from the parameter YAML)"
        ),
    )
    parser.add_argument(
        "--stage3-integration-dt-ms",
        type=float,
        default=None,
        help=(
            "Stage 3-only integration step in ms; benchmark-only override "
            "(default: value from the parameter YAML)"
        ),
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup_repeats < 0:
        parser.error("--repeats must be positive and --warmup-repeats must be non-negative")
    if args.outgoing_velocity_tolerance_mm is not None and (
        not math.isfinite(args.outgoing_velocity_tolerance_mm)
        or args.outgoing_velocity_tolerance_mm <= 0.0
    ):
        parser.error("--outgoing-velocity-tolerance-mm must be finite and positive")
    if args.integration_dt_ms is not None and (
        not math.isfinite(args.integration_dt_ms) or args.integration_dt_ms <= 0.0
    ):
        parser.error("--integration-dt-ms must be finite and positive")
    if args.stage3_integration_dt_ms is not None and (
        not math.isfinite(args.stage3_integration_dt_ms)
        or args.stage3_integration_dt_ms <= 0.0
    ):
        parser.error("--stage3-integration-dt-ms must be finite and positive")
    return args


def main() -> None:
    args = _parse_args()
    trace_path = resolve_ball_trace(args.trace)
    params_path = args.params_file or _default_params_path()
    if not params_path.is_file():
        raise FileNotFoundError(f"parameter file not found: {params_path}")
    config, physics = load_planner_settings(params_path)
    if args.integration_dt_ms is not None:
        config.dt_integrate = args.integration_dt_ms * 1e-3
    if args.stage3_integration_dt_ms is not None:
        config.stage3_dt_integrate = args.stage3_integration_dt_ms * 1e-3
    samples, skipped = load_ball_samples(trace_path)

    print(f"trace: {trace_path} ({len(samples)} finite samples; skipped={skipped})")
    print(f"params: {params_path}")
    print(
        "mode: "
        + ("collect prediction samples (no CSV I/O)" if args.collect_prediction_samples
           else "normal non-recording")
    )
    tolerance_m = (
        args.outgoing_velocity_tolerance_mm * 1e-3
        if args.outgoing_velocity_tolerance_mm is not None
        else config.outgoing_velocity_tolerance_m
    )
    print(f"outgoing-velocity landing tolerance: {tolerance_m * 1e3:g} mm")
    print(f"integration dt: {config.dt_integrate * 1e3:g} ms")
    print(f"Stage 3 integration dt: {config.stage3_dt_integrate * 1e3:g} ms")

    for _ in range(args.warmup_repeats):
        benchmark_once(
            samples, config, physics, args.collect_prediction_samples, tolerance_m
        )

    for repeat in range(1, args.repeats + 1):
        frame_summary, planning_summary, stage_summaries, published = benchmark_once(
            samples, config, physics, args.collect_prediction_samples, tolerance_m
        )
        print(f"run {repeat}/{args.repeats}; returned commands={published}")
        print(_format_summary("all update() frames", frame_summary))
        print(_format_summary("full Stage 1->2->3 frames", planning_summary))
        print(_format_summary("  Stage 1 (buffer + estimate)", stage_summaries["stage1"]))
        print(_format_summary("  Stage 2 (trajectory prediction)", stage_summaries["stage2"]))
        print(_format_summary("  Stage 3 (racket target)", stage_summaries["stage3"]))
        print(_format_summary("  other pipeline overhead", stage_summaries["other"]))


if __name__ == "__main__":
    main()

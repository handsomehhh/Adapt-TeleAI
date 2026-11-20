"""Top-level HOPE planner pipeline combining Stages 1-3.

Call .update() with each ball position at the motion capture frame rate;
read .racket_command for the latest desired racket state.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 6.
"""

from typing import Optional
import time

import numpy as np

from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor, StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams
from .racket_target_planner import RacketCommand, RacketTargetPlanner


use_print = False  # TODO: remove debug prints from planner.py


class HOPEPlanner:
    """Top-level planner combining Stages 1-3."""

    def __init__(
        self,
        physics: Optional[BallPhysics] = None,
        config: Optional[PlannerConfig] = None,
        table: Optional[TableParams] = None,
    ):

        self.count_ball_number = 0

        self.physics = physics or BallPhysics()
        self.config = config or PlannerConfig()
        self.table = table or TableParams()

        self.estimator = BallStateEstimator(self.config)
        self.predictor = BallTrajectoryPredictor(self.physics, self.config, self.table)
        self.target_planner = RacketTargetPlanner(
            self.physics,
            self.config,
            self.table,
            outgoing_velocity_tolerance_m=self.config.outgoing_velocity_tolerance_m,
        )

        self._latest_command: Optional[RacketCommand] = None
        self._latest_strike: Optional[StrikeTarget] = None
        self._latest_t: Optional[float] = None
        self._latest_estimate: Optional[np.ndarray] = None
        self._latest_velocity: Optional[np.ndarray] = None
        self._latest_prediction_samples: Optional[np.ndarray] = None
        # Filled only when update(profile_stages=True) is requested by the
        # offline benchmark.  Normal ROS operation does not call the clock.
        self._last_stage_timings_ns: dict[str, int] = {
            "stage1": 0,
            "stage2": 0,
            "stage3": 0,
        }
        # Per-frame, read-only diagnostics consumed by the replay visualizer.
        # Keep this independent of ROS so offline callers see identical facts.
        self._last_update_diagnostics: dict[str, object] = {
            "buffer_points": 0,
            "buffer_min_fit_samples": self.config.min_fit_samples,
            "current_point_pushed": False,
            "current_point_reason": "no input received yet",
            "buffer_reset_reason": "",
            "command_published": False,
            "command_reason": "no input received yet",
            "estimated_vx_m_s": None,
            # Stage 3 can produce a numerically valid target which is then
            # intentionally held back by the publish-workspace gate.  Keep
            # that candidate in diagnostics so operators can distinguish it
            # from the previously published command shown by a subscriber.
            "candidate_intercept_x_m": None,
            "candidate_intercept_y_m": None,
            "candidate_intercept_z_m": None,
            "candidate_velocity_x_m_s": None,
            "candidate_velocity_y_m_s": None,
            "candidate_velocity_z_m_s": None,
            "candidate_time_to_strike_s": None,
            "candidate_valid": None,
        }

    def _frame_diagnostics(
        self,
        *,
        point_pushed: bool,
        point_reason: str,
        command_published: bool,
        command_reason: str,
        estimated_vx: Optional[float] = None,
        candidate_command: Optional[RacketCommand] = None,
        candidate_time_to_strike_s: Optional[float] = None,
    ) -> None:
        """Capture the exact decision path of the latest input frame."""
        cleared = self.estimator.cleared_reason or ""
        candidate = {}
        if candidate_command is not None:
            candidate = {
                "candidate_intercept_x_m": float(candidate_command.p_intercept[0]),
                "candidate_intercept_y_m": float(candidate_command.p_intercept[1]),
                "candidate_intercept_z_m": float(candidate_command.p_intercept[2]),
                "candidate_velocity_x_m_s": float(candidate_command.v_racket[0]),
                "candidate_velocity_y_m_s": float(candidate_command.v_racket[1]),
                "candidate_velocity_z_m_s": float(candidate_command.v_racket[2]),
                "candidate_time_to_strike_s": candidate_time_to_strike_s,
                "candidate_valid": bool(candidate_command.valid),
            }
        self._last_update_diagnostics = {
            "buffer_points": len(self.estimator.t_buffer),
            "buffer_min_fit_samples": self.config.min_fit_samples,
            "current_point_pushed": point_pushed,
            "current_point_reason": point_reason,
            "buffer_reset_reason": cleared,
            "command_published": command_published,
            "command_reason": command_reason,
            "estimated_vx_m_s": estimated_vx,
            "candidate_intercept_x_m": None,
            "candidate_intercept_y_m": None,
            "candidate_intercept_z_m": None,
            "candidate_velocity_x_m_s": None,
            "candidate_velocity_y_m_s": None,
            "candidate_velocity_z_m_s": None,
            "candidate_time_to_strike_s": None,
            "candidate_valid": None,
            **candidate,
        }

    def note_input_not_pushed(self, reason: str) -> None:
        """Expose a node-side rejected input without mutating estimator state."""
        self._latest_command = None
        self._latest_strike = None
        self._frame_diagnostics(
            point_pushed=False,
            point_reason=reason,
            command_published=False,
            command_reason=f"input_not_processed:{reason}",
        )

    def update(
        self,
        t: float,
        p_ball: np.ndarray,
        *,
        collect_prediction_samples: bool = False,
        profile_stages: bool = False,
    ) -> Optional[RacketCommand]:
        """Process a new ball position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic).
        p_ball : np.ndarray, shape (3,)
            Ball position [x, y, z] in HOPE canonical frame.
        collect_prediction_samples : bool, optional
            Collect the full 1 ms prediction path for recording/replay.  This
            is disabled by default because command generation only needs the
            strike target, not every intermediate integration point.
        profile_stages : bool, optional
            Measure the Stage 1/2/3 portions of this update.  Intended only
            for the offline timing benchmark; disabled by default so ROS
            operation does not pay for clock reads.

        Returns
        -------
        RacketCommand or None
        """
        if p_ball[0] > 0.0 and p_ball[0] < 2.5 and p_ball[1] > -1.6 and p_ball[1] < 0.0 and p_ball[2] > 0.0 and p_ball[2] < 1.0:
            self.count_ball_number += 1
            if use_print:
                print("timestamp: ", t, "ball number: ", self.count_ball_number, " p_ball: ", p_ball)
        # else: # 不应该在这里return none，会无法清空buffer
        #     return None

        if profile_stages:
            self._last_stage_timings_ns = {"stage1": 0, "stage2": 0, "stage3": 0}
            stage1_start_ns = time.perf_counter_ns()
        point_pushed = self.estimator.push(t, p_ball)
        reset_reason = self.estimator.cleared_reason
        if reset_reason:
            # A bounce, direction reversal, or workspace reset starts a new
            # flight leg.  Do not use the previous leg's Stage 3 warm start.
            self.target_planner.clear_outgoing_velocity_cache()
        point_reason = (
            f"pushed_after_buffer_reset:{reset_reason}"
            if point_pushed and reset_reason else
            "pushed" if point_pushed else
            f"not_pushed:{reset_reason or 'rejected'}"
        )

        if not self.estimator.ready:
            self._latest_command = None
            self._latest_strike = None
            self._frame_diagnostics(
                point_pushed=point_pushed,
                point_reason=point_reason,
                command_published=False,
                command_reason=(
                    "buffer_not_ready:"
                    f"{len(self.estimator.t_buffer)}/{self.config.min_fit_samples}"
                ),
            )
            return None

        # 刚才30 - 6 的时候，能进入下面的打印，说明buffer里至少有6个点了，但是仍然预测出速度为正的时候？
        p_est, v_est, t_est = self.estimator.estimate()
        if profile_stages:
            self._last_stage_timings_ns["stage1"] = (
                time.perf_counter_ns() - stage1_start_ns
            )
        stage1_fit_rmse_m = self.estimator.fit_rmse_m
        self._latest_t = t_est
        self._latest_estimate = p_est.copy()
        self._latest_velocity = v_est.copy()

        # Only predict if the ball is moving toward P1 (v_x < 0).
        if v_est[0] > 0:
            # if v_est[0] > 0.001:
            #     print("vx: ",v_est[0]," p_ball_x: ",p_ball[0]," p_est_x: ",p_est[0])
            #     # 同时打印出buffer中的所有p_buffer[:,0]
            #     print("len:",len(self.estimator.p_buffer),"  p buffer: ",[x for x in self.estimator.p_buffer], "  t buffer: ", [x for x in self.estimator.t_buffer])
            #     print("v_est > 0.001 !!!!!!!")
            #     print()
            self._latest_command = None
            self._latest_strike = None
            if use_print:
                print("None because v_est[0] > 0.001, v_est[0]: ", v_est[0], " p_ball_x: ", p_ball[0], " p_est_x: ", p_est[0])
            self._frame_diagnostics(
                point_pushed=True,
                point_reason=point_reason,
                command_published=False,
                command_reason="estimated_vx_positive_moving_away_from_P1",
                estimated_vx=float(v_est[0]),
            )
            return None
    

        # Suppress time-to-strike and racket targets while the incoming ball is
        # still too far on the opponent side. This keeps early inbound samples
        # from producing actionable strike timing.
        if p_ball[0] > self.config.time_to_strike_invalid_x:
            self._latest_command = None
            self._latest_strike = None
            if use_print:
                print("None because p_ball[0] > self.config.time_to_strike_invalid_x, p_ball[0]: ", p_ball[0], " self.config.time_to_strike_invalid_x: ", self.config.time_to_strike_invalid_x)
            self._frame_diagnostics(
                point_pushed=True,
                point_reason=point_reason,
                command_published=False,
                command_reason=(
                    "input_x_above_time_to_strike_limit:"
                    f"{p_ball[0]:.4f}>{self.config.time_to_strike_invalid_x:.4f}"
                ),
                estimated_vx=float(v_est[0]),
            )
            return None

        if profile_stages:
            stage2_start_ns = time.perf_counter_ns()
        if collect_prediction_samples:
            strike, prediction_samples = self.predictor.predict_with_samples(
                p_est, v_est, t_est
            )
        else:
            strike = self.predictor.predict(p_est, v_est, t_est)
            prediction_samples = None
        if profile_stages:
            self._last_stage_timings_ns["stage2"] = (
                time.perf_counter_ns() - stage2_start_ns
            )
        self._latest_strike = strike
        self._latest_prediction_samples = prediction_samples

        if profile_stages:
            stage3_start_ns = time.perf_counter_ns()
        command = self.target_planner.plan(strike)
        if profile_stages:
            self._last_stage_timings_ns["stage3"] = (
                time.perf_counter_ns() - stage3_start_ns
            )
        # Carry the fit quality alongside the command so downstream consumers
        # receive the exact Stage 1 result that informed this plan.
        command.stage1_fit_rmse_m = float(stage1_fit_rmse_m)
        # Do not hand an unreachable intercept to any downstream publisher.
        # This gate belongs in the core pipeline (rather than only in the ROS
        # node) so offline and alternate callers observe the same contract.
        if not self.config.allows_publish_intercept(command.p_intercept):
            if use_print:
                print("None because not self.config.allows_publish_intercept(command.p_intercept), command.p_intercept: ", command.p_intercept)
            self._latest_command = None
            self._frame_diagnostics(
                point_pushed=True,
                point_reason=point_reason,
                command_published=False,
                command_reason="intercept_outside_publish_workspace",
                estimated_vx=float(v_est[0]),
                candidate_command=command,
                candidate_time_to_strike_s=float(command.t_strike - t_est),
            )
            return None
        self._latest_command = command
        self._frame_diagnostics(
            point_pushed=True,
            point_reason=point_reason,
            command_published=True,
            command_reason=(
                "published_valid_command" if command.valid
                else "published_invalid_command:no_hit_plane_crossing"
            ),
            estimated_vx=float(v_est[0]),
            candidate_command=command,
            candidate_time_to_strike_s=float(command.t_strike - t_est),
        )
        return command

    @property
    def racket_command(self) -> Optional[RacketCommand]:
        return self._latest_command

    @property
    def time_to_strike(self) -> Optional[float]:
        """Seconds remaining until the predicted strike (positive, decreasing).

        NOTE: this returns time *remaining* (t_strike - latest sample time),
        not the absolute strike time. The reference doc skeleton returned the
        absolute time; that is corrected here so the value is positive and
        decreases as the ball approaches, per the verification gate.
        """
        if self._latest_command is None or not self._latest_command.valid:
            return None
        if self._latest_strike is None or self._latest_t is None:
            return None
        return self._latest_strike.t_strike - self._latest_t

    @property
    def latest_estimate(self) -> Optional[np.ndarray]:
        return None if self._latest_estimate is None else self._latest_estimate.copy()

    @property
    def latest_velocity(self) -> Optional[np.ndarray]:
        return None if self._latest_velocity is None else self._latest_velocity.copy()

    @property
    def latest_prediction_samples(self) -> Optional[np.ndarray]:
        """Rows ``[prediction_time, x, y, z]`` from the exact planner loop."""
        return (None if self._latest_prediction_samples is None
                else self._latest_prediction_samples.copy())

    @property
    def last_update_diagnostics(self) -> dict[str, object]:
        """Decision facts from the most recently received input frame."""
        return self._last_update_diagnostics.copy()

    @property
    def last_stage_timings_ns(self) -> dict[str, int]:
        """Optional Stage 1/2/3 timing from the most recently profiled update."""
        return self._last_stage_timings_ns.copy()

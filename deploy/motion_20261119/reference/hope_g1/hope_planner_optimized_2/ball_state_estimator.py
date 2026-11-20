"""Stage 1 - Ball state estimation.

Fits a 2nd-order polynomial to the most recent N position samples and
differentiates analytically to obtain a smoothed position and velocity.
The buffer is cleared so the polynomial never fits across a velocity
discontinuity or an unrelated flight leg, in three situations:

  1. a detected table bounce (descend -> contact -> rise pattern);
  2. the ball is flying in the +x direction (x velocity > 0), i.e. it is
     travelling away from P1 rather than inbound;
  3. the ball has travelled past ``config.clear_x_max`` (x > 2.5 m).

Velocity is only computed once the buffer has accumulated at least
``config.min_fit_samples`` frames.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 3.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import PlannerConfig


class BallStateEstimator:
    """Estimate ball position and velocity from a position stream.

    Maintains a sliding window of recent position measurements and performs
    a least-squares polynomial fit to extract smoothed position and velocity.

    Bounce detection uses a three-sample pattern (descend -> contact -> rise)
    to identify the actual table impact event and clear the buffer. The buffer
    is also cleared when the ball is flying in the +x direction or has passed
    ``config.clear_x_max`` (see module docstring).
    """

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.t_buffer: List[float] = []
        self.p_buffer: List[np.ndarray] = []

        # Bounce detection: three-sample z-height ring buffer.
        # Initialized to None to suppress false triggers before
        # enough measurements are collected.
        self._z_hist: List[Optional[float]] = [None, None, None]
        self._bounce_detected: bool = False
        # Reason the buffer was cleared on the most recent push(), or None.
        # One of: "bounce", "vx_positive", "x_beyond_max".
        self._cleared_reason: Optional[str] = None
        # 3D position RMSE (m) of the most recently computed polynomial fit.
        # It is reset whenever the sample buffer is reset, so callers cannot
        # accidentally associate a fit quality value with a different flight.
        self._fit_rmse_m: Optional[float] = None

    def reset(self) -> None:
        """Clear the estimation buffer (called on bounce / clear conditions)."""
        self.t_buffer.clear()
        self.p_buffer.clear()
        self._fit_rmse_m = None

    def push(self, t: float, p: np.ndarray) -> bool:
        """Add a new position measurement and report that it was buffered.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic, e.g. from ROS clock).
        p : np.ndarray, shape (3,)
            Ball position [x, y, z] in the HOPE canonical frame.

        修改：这个点不合法，就直接不要推进buffer里
        """
        # Update z history ring buffer
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = p[2]

        self._bounce_detected = False
        self._cleared_reason = None

        # Bounce detection: three-sample pattern.
        # z_prev_prev was above table, z_prev dipped to contact,
        # z_curr is rising again -> actual bounce event.
        # Clear before buffering the current point, so the first observation of
        # the new flight leg is retained as the new buffer's first point.
        z_pp, z_p, z_c = self._z_hist
        tol = self.config.bounce_z_tol
        if z_pp is not None and z_p is not None and z_c is not None:
            # print(z_pp, z_p, z_c)
            # if z_pp > tol and z_p <= tol and z_c > tol:
            if (z_pp - z_p > 0.01 and z_c - z_p > 0.01) or (z_c <= tol):
                self._bounce_detected = True
                self._cleared_reason = "bounce"
                self.reset()

        # Clear condition: ball flying in the +x direction (moving away from
        # P1). Use a finite difference against the previous sample so a reversal
        # resets the buffer immediately, without waiting for the polynomial fit.



        # Clear condition: ball has travelled past the far cutoff.
        # x=0 is the configured hit plane and must remain a valid input point.
        if p[0] > self.config.clear_x_max or p[0] < 0.0 or p[1] > 0.0 or p[1] < -1.5 or p[2] < 0.0:
            # if p[0] < 2.7 and p[0] > 0.0:
            # print("x_beyond_max: ",p[0])
            self._cleared_reason = "x_beyond_max"
            self.reset()

        # 改为用位置判断 -- 出现明显较大误差的反向（不是噪声）
        if self.t_buffer:
            dt = t - self.t_buffer[-1]
            if dt > 0.0 and (p[0] - self.p_buffer[-1][0]) > 0.001 : # 1mm
                # print("vx: ",(p[0] - self.p_buffer[-1][0]) / dt," p_ball_x: ",p[0]," last p x: ",self.p_buffer[-1][0])
                self._cleared_reason = "vx_positive"
                self.reset()
        

        # 如果满足上述条件，但是新的点和旧的点相差太大，空但是push
        # x方向距离过大？还是时间戳过大？



        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        if len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)
        return True

    @property
    def bounce_detected(self) -> bool:
        """True if the most recent push() detected a table bounce."""
        return self._bounce_detected

    @property
    def cleared_reason(self) -> Optional[str]:
        """Why the buffer was cleared on the last push(), else None.

        One of ``"bounce"``, ``"vx_positive"``, ``"x_beyond_max"``.
        """
        return self._cleared_reason

    @property
    def ready(self) -> bool:
        """True if enough samples exist for a reliable fit."""
        return len(self.t_buffer) >= self.config.min_fit_samples

    @property
    def fit_rmse_m(self) -> Optional[float]:
        """3D position RMSE (m) of the most recent Stage 1 polynomial fit.

        The residual for one sample is the Euclidean distance between its
        measured position and the fitted three-dimensional trajectory.  The
        stored value is therefore in metres rather than metres squared.
        """
        return self._fit_rmse_m

    def estimate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute smoothed ball position and velocity at the latest timestamp.

        Returns
        -------
        p_est : np.ndarray, shape (3,)
            Smoothed position estimate [x, y, z] in HOPE frame.
        v_est : np.ndarray, shape (3,)
            Velocity estimate [vx, vy, vz] in HOPE frame (m/s).
        t_est : float
            Timestamp of the estimate (latest sample time).
        """
        if not self.ready:
            raise RuntimeError(
                f"Need >= {self.config.min_fit_samples} samples, "
                f"have {len(self.t_buffer)}"
            )

        t_arr = np.array(self.t_buffer)
        p_arr = np.array(self.p_buffer)

        # Normalize time to improve numerical conditioning
        t_ref = t_arr[-1]
        t_norm = t_arr - t_ref

        p_est = np.zeros(3)
        v_est = np.zeros(3)
        p_fit = np.empty(p_arr.shape, dtype=float)

        for axis in range(3):
            # np.polyfit returns [a2, a1, a0] (descending order)
            coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=self.config.poly_order)
            p_est[axis] = coeffs[-1]   # a0 at t_norm = 0
            v_est[axis] = coeffs[-2]   # a1 at t_norm = 0
            p_fit[:, axis] = np.polyval(coeffs, t_norm)

        residual_norms = np.linalg.norm(p_arr - p_fit, axis=1)
        self._fit_rmse_m = float(np.sqrt(np.mean(residual_norms ** 2)))

        return p_est, v_est, t_ref

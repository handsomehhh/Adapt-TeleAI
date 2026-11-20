"""Stage 2 - Ball trajectory prediction.

Forward-integrate the ball trajectory with explicit Euler at 1 kHz using a
hybrid flight (quadratic drag + gravity) / bounce (diagonal restitution)
model, and return the predicted ball state at the virtual hitting plane.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 4.
"""

from dataclasses import dataclass
import math

import numpy as np

from .constants import BallPhysics, PlannerConfig, TableParams


@dataclass
class StrikeTarget:
    """Output of Stage 2: predicted ball state at the hitting plane."""

    p_ball: np.ndarray        # predicted ball position at strike [x, y, z]
    v_ball: np.ndarray        # predicted ball velocity at strike [vx, vy, vz]
    t_strike: float           # absolute time of strike
    num_bounces: int          # number of table bounces before strike
    valid: bool               # True if a valid crossing was found


class BallTrajectoryPredictor:
    """Forward-integrate ball trajectory and find the hitting-plane crossing.

    Uses explicit Euler integration with the hybrid flight/bounce model
    from HITTER Section III-B.
    """

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _is_on_table(self, p: np.ndarray) -> bool:
        """Check if the ball could contact the table surface.

        Bounds are expanded by ball radius to handle edge contacts.
        """
        r = self.physics.radius
        return (
            -r <= p[0] <= self.table.length + r
            and -self.table.width - r <= p[1] <= r
        )

    def _flight_acceleration(self, v: np.ndarray) -> np.ndarray:
        """Compute ball acceleration during free flight: a = -k|v|v + g"""
        speed = np.linalg.norm(v)
        return -self.physics.k * speed * v + self.physics.g

    def _apply_bounce(self, v: np.ndarray) -> np.ndarray:
        """Apply table bounce restitution: v+ = diag(C_h, C_h, -C_v) @ v-"""
        C = np.diag([self.physics.C_h, self.physics.C_h, -self.physics.C_v])
        return C @ v

    def predict(self, p0: np.ndarray, v0: np.ndarray, t0: float) -> StrikeTarget:
        """Return the interception state used by the planner."""
        return self._predict(p0, v0, t0, samples=None)

    def predict_with_samples(
        self, p0: np.ndarray, v0: np.ndarray, t0: float,
    ) -> tuple[StrikeTarget, np.ndarray]:
        """Return the normal prediction plus its exact sampled integration path.

        This is intentionally backed by the same integration loop as
        :meth:`predict`; it exists for planner-side recording and must not
        create a second, subtly different visualisation model.
        """
        samples: list[list[float]] = []
        strike = self._predict(p0, v0, t0, samples=samples)
        return strike, np.asarray(samples, dtype=float)

    def _predict(
        self, p0: np.ndarray, v0: np.ndarray, t0: float,
        samples: list[list[float]] | None,
    ) -> StrikeTarget:
        """Forward-integrate and find the hitting-plane crossing.

        Parameters
        ----------
        p0 : Current ball position in HOPE frame.
        v0 : Current ball velocity in HOPE frame.
        t0 : Current timestamp (s).

        Returns
        -------
        StrikeTarget with predicted ball state at the virtual hitting plane.
        """
        dt = self.config.dt_integrate
        max_steps = int(self.config.max_predict_time / dt)
        x_hit = self.config.x_hit

        # The integration runs at 1 kHz.  Keep the state in Python scalars so
        # this hot loop does not allocate several tiny NumPy arrays or dispatch
        # np.linalg.norm on every step.  Arrays are constructed only for the
        # final StrikeTarget, preserving the public API.
        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        t = t0
        bounces = 0
        k = self.physics.k
        gx, gy, gz = (
            float(self.physics.g[0]),
            float(self.physics.g[1]),
            float(self.physics.g[2]),
        )
        contact_z = self.physics.radius
        dt_sq = dt ** 2
        if samples is not None:
            samples.append([t, x, y, z])

        for _step in range(max_steps):
            p_prev_x = x

            # --- Euler integration step ---
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            drag = -k * speed
            ax, ay, az = drag * vx + gx, drag * vy + gy, drag * vz + gz
            vx_new, vy_new, vz_new = vx + ax * dt, vy + ay * dt, vz + az * dt
            x_new = x + vx * dt + 0.5 * ax * dt_sq
            y_new = y + vy * dt + 0.5 * ay * dt_sq
            z_new = z + vz * dt + 0.5 * az * dt_sq
            t += dt
            bounce_this_step = False

            # --- Bounce detection ---
            # p is the ball center, so table contact happens when the center
            # reaches one ball radius above the table plane.
            if z_new < contact_z and vz_new < 0.0:
                # Sub-step interpolation to find exact bounce time.
                dz = z - z_new
                frac = (z - contact_z) / dz if dz > 1e-9 else 0.5
                frac = min(1.0, max(0.0, frac))

                x_bounce = x + frac * (x_new - x)
                y_bounce = y + frac * (y_new - y)
                z_bounce = contact_z
                radius = self.physics.radius
                on_table = (
                    -radius <= x_bounce <= self.table.length + radius
                    and -self.table.width - radius <= y_bounce <= radius
                )
                if on_table:
                    bounce_dt = frac * dt
                    vx_at = vx + ax * bounce_dt
                    vy_at = vy + ay * bounce_dt
                    vz_at = vz + az * bounce_dt

                    vx_post = self.physics.C_h * vx_at
                    vy_post = self.physics.C_h * vy_at
                    vz_post = -self.physics.C_v * vz_at

                    # Continue from bounce with second-order correction
                    remaining_dt = (1.0 - frac) * dt
                    speed_post = math.sqrt(
                        vx_post * vx_post + vy_post * vy_post + vz_post * vz_post
                    )
                    drag_post = -k * speed_post
                    ax_post = drag_post * vx_post + gx
                    ay_post = drag_post * vy_post + gy
                    az_post = drag_post * vz_post + gz
                    remaining_dt_sq = remaining_dt ** 2
                    x_new = (
                        x_bounce + vx_post * remaining_dt
                        + 0.5 * ax_post * remaining_dt_sq
                    )
                    y_new = (
                        y_bounce + vy_post * remaining_dt
                        + 0.5 * ay_post * remaining_dt_sq
                    )
                    z_new = (
                        z_bounce + vz_post * remaining_dt
                        + 0.5 * az_post * remaining_dt_sq
                    )
                    vx_new = vx_post + ax_post * remaining_dt
                    vy_new = vy_post + ay_post * remaining_dt
                    vz_new = vz_post + az_post * remaining_dt
                    bounces += 1
                    bounce_this_step = True
                else:
                    z_new = max(z_new, contact_z)

            # --- Hitting plane crossing detection ---
            if p_prev_x > x_hit and x_new <= x_hit and vx_new < 0:
                if bounce_this_step:
                    # Use post-bounce arc for interpolation
                    dx_arc = x_bounce - x_new
                    if abs(dx_arc) > 1e-9:
                        frac_cross = (x_bounce - x_hit) / dx_arc
                    else:
                        frac_cross = 0.5
                    frac_cross = min(1.0, max(0.0, frac_cross))
                    y_cross = y_bounce + frac_cross * (y_new - y_bounce)
                    z_cross = z_bounce + frac_cross * (z_new - z_bounce)
                    vx_cross = vx_post + frac_cross * (vx_new - vx_post)
                    vy_cross = vy_post + frac_cross * (vy_new - vy_post)
                    vz_cross = vz_post + frac_cross * (vz_new - vz_post)
                    t_cross = (t - remaining_dt) + frac_cross * remaining_dt
                else:
                    dx_step = x - x_new
                    if abs(dx_step) > 1e-9:
                        frac_cross = (x - x_hit) / dx_step
                    else:
                        frac_cross = 0.5
                    frac_cross = min(1.0, max(0.0, frac_cross))
                    y_cross = y + frac_cross * (y_new - y)
                    z_cross = z + frac_cross * (z_new - z)
                    vx_cross = vx + frac_cross * (vx_new - vx)
                    vy_cross = vy + frac_cross * (vy_new - vy)
                    vz_cross = vz + frac_cross * (vz_new - vz)
                    t_cross = t - dt + frac_cross * dt

                if samples is not None:
                    samples.append([t_cross, x_hit, y_cross, z_cross])

                return StrikeTarget(
                    p_ball=np.array([x_hit, y_cross, z_cross]),
                    v_ball=np.array([vx_cross, vy_cross, vz_cross]),
                    t_strike=t_cross, num_bounces=bounces, valid=True,
                )

            x, y, z = x_new, y_new, z_new
            vx, vy, vz = vx_new, vy_new, vz_new
            if samples is not None:
                samples.append([t, x, y, z])

        return StrikeTarget(
            p_ball=np.array([x, y, z]),
            v_ball=np.array([vx, vy, vz]),
            t_strike=t,
            num_bounces=bounces, valid=False,
        )

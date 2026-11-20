"""Stage 2 - Ball trajectory prediction.

Forward-integrate the ball trajectory with explicit Euler at 1 kHz using a
hybrid flight (quadratic drag + gravity) / bounce (diagonal restitution)
model, and return the predicted ball state at the virtual hitting plane.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 4.
"""

from dataclasses import dataclass

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

        p = p0.copy()
        v = v0.copy()
        t = t0
        bounces = 0
        if samples is not None:
            samples.append([t, p[0], p[1], p[2]])

        # Track the most recent bounce state so a hit-plane crossing that
        # happens in the same step as a bounce interpolates on a continuous arc.
        p_bounce = p.copy()
        v_post = v.copy()
        remaining_dt = dt

        for _step in range(max_steps):
            p_prev_x = p[0]

            # --- Euler integration step ---
            a = self._flight_acceleration(v)
            v_new = v + a * dt
            p_new = p + v * dt + 0.5 * a * dt ** 2
            t += dt
            bounce_this_step = False

            # --- Bounce detection ---
            # p is the ball center, so table contact happens when the center
            # reaches one ball radius above the table plane.
            contact_z = self.physics.radius
            if p_new[2] < contact_z and v_new[2] < 0.0:
                # Sub-step interpolation to find exact bounce time.
                dz = p[2] - p_new[2]
                frac = (p[2] - contact_z) / dz if dz > 1e-9 else 0.5
                frac = np.clip(frac, 0.0, 1.0)

                p_bounce = p + frac * (p_new - p)
                p_bounce[2] = contact_z
                if self._is_on_table(p_bounce):
                    v_at_bounce = v + a * (frac * dt)

                    v_post = self._apply_bounce(v_at_bounce)

                    # Continue from bounce with second-order correction
                    remaining_dt = (1.0 - frac) * dt
                    a_post = self._flight_acceleration(v_post)
                    p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt ** 2
                    v_new = v_post + a_post * remaining_dt
                    bounces += 1
                    bounce_this_step = True
                else:
                    p_new[2] = max(p_new[2], contact_z)

            # --- Hitting plane crossing detection ---
            if p_prev_x > x_hit and p_new[0] <= x_hit and v_new[0] < 0:
                if bounce_this_step:
                    # Use post-bounce arc for interpolation
                    dx_arc = p_bounce[0] - p_new[0]
                    if abs(dx_arc) > 1e-9:
                        frac_cross = (p_bounce[0] - x_hit) / dx_arc
                    else:
                        frac_cross = 0.5
                    frac_cross = np.clip(frac_cross, 0.0, 1.0)
                    p_cross = p_bounce + frac_cross * (p_new - p_bounce)
                    v_cross = v_post + frac_cross * (v_new - v_post)
                    t_cross = (t - remaining_dt) + frac_cross * remaining_dt
                else:
                    dx_step = p[0] - p_new[0]
                    if abs(dx_step) > 1e-9:
                        frac_cross = (p[0] - x_hit) / dx_step
                    else:
                        frac_cross = 0.5
                    frac_cross = np.clip(frac_cross, 0.0, 1.0)
                    p_cross = p + frac_cross * (p_new - p)
                    v_cross = v + frac_cross * (v_new - v)
                    t_cross = t - dt + frac_cross * dt

                p_cross[0] = x_hit
                if samples is not None:
                    samples.append([t_cross, p_cross[0], p_cross[1], p_cross[2]])

                return StrikeTarget(
                    p_ball=p_cross, v_ball=v_cross,
                    t_strike=t_cross, num_bounces=bounces, valid=True,
                )

            p = p_new
            v = v_new
            if samples is not None:
                samples.append([t, p[0], p[1], p[2]])

        return StrikeTarget(
            p_ball=p, v_ball=v, t_strike=t,
            num_bounces=bounces, valid=False,
        )

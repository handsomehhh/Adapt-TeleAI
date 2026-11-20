"""Stage 3 - Racket target planning.

Given the predicted ball state at the hitting plane (Stage 2), compute the
desired racket velocity and face orientation to return the ball to the
opponent's half center, with a net-clearance check and flight-time fallback.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 5.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from .ball_trajectory_predictor import StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams


@dataclass
class RacketCommand:
    """Output of Stage 3: desired racket state at strike time.

    This is the planner's output to the whole-body controller. The racket's
    actual pose is inferred by the humanoid via forward kinematics - it is
    never measured by the motion capture system.
    """

    p_intercept: np.ndarray   # desired racket center position at interception
    v_racket: np.ndarray      # desired racket velocity vector [vx, vy, vz]
    n_racket: np.ndarray      # desired racket face normal (unit vector)
    t_strike: float           # predicted time of strike
    v_ball_outgoing: np.ndarray  # expected outgoing ball velocity
    target_land: np.ndarray   # intended landing point
    clears_net: bool          # True if return trajectory clears the net
    bypasses_net_posts: bool  # True if ball passes outside net Y extent
    valid: bool               # True if all computations succeeded
    num_bounces: int = 0      # bounces predicted before the strike (from Stage 2)


class RacketTargetPlanner:
    """Compute desired racket velocity and orientation for a valid return.

    Implements HITTER Section III-C: racket-ball interaction model.
    """

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _compute_outgoing_velocity(
        self, p_strike: np.ndarray, p_land: np.ndarray, delta_t: float,
    ) -> np.ndarray:
        """Solve post-strike velocity with a finite-difference Newton method.

        The original planner used 24 fixed-point iterations.  Each iteration
        ran a 1 kHz Python trajectory simulation, making this the dominant
        real-time cost.  Here, the same flight model is evaluated with scalar
        arithmetic and Newton's method.  It normally converges in 3--4
        iterations, while retaining a fixed-point fallback for singular or
        unusual trajectories.
        """
        if delta_t <= 1e-6:
            raise ValueError("delta_t must be positive")

        # p = p0 + v*dt + 0.5*g*dt^2  =>  v = (p-p0)/dt - 0.5*g*dt
        v = (p_land - p_strike) / delta_t - 0.5 * self.physics.g * delta_t
        jacobian_step = 5e-3
        for _ in range(6):
            p_end = self._integrate_position_scalar(p_strike, v, delta_t)
            residual = p_end - p_land
            if float(np.dot(residual, residual)) < 1e-8:
                return v

            jacobian = np.empty((3, 3))
            for axis in range(3):
                perturbed = v.copy()
                perturbed[axis] += jacobian_step
                jacobian[:, axis] = (
                    self._integrate_position_scalar(p_strike, perturbed, delta_t)
                    - p_end
                ) / jacobian_step
            try:
                correction = np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(correction)):
                break
            v = v - correction

        # Preserve a robust fallback for pathological geometry.  This path is
        # uncommon, and still uses the scalar integrator below.
        for _ in range(12):
            p_end = self._integrate_position_scalar(p_strike, v, delta_t)
            error = p_land - p_end
            if float(np.dot(error, error)) < 1e-8:
                break
            v = v + error / delta_t
        return v

    def _flight_acceleration(self, v: np.ndarray) -> np.ndarray:
        """Compute free-flight acceleration: a = -k|v|v + g."""
        speed = np.linalg.norm(v)
        return -self.physics.k * speed * v + self.physics.g

    def _integrate_flight(
        self, p0: np.ndarray, v0: np.ndarray, duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate free flight with the same model used by Stage 2.

        The integration loop deliberately uses Python scalars: repeatedly
        allocating tiny NumPy arrays and dispatching ``np.linalg.norm`` is far
        more expensive than the arithmetic for a 3-D state.
        """
        x, y, z, vx, vy, vz = self._integrate_flight_scalar(p0, v0, duration)
        return np.array([x, y, z]), np.array([vx, vy, vz])

    def _integrate_position_scalar(
        self, p0: np.ndarray, v0: np.ndarray, duration: float,
    ) -> np.ndarray:
        """Return only final position, avoiding velocity-array allocation."""
        x, y, z, _, _, _ = self._integrate_flight_scalar(p0, v0, duration)
        return np.array([x, y, z])

    def _integrate_flight_scalar(
        self, p0: np.ndarray, v0: np.ndarray, duration: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Scalar version of the original second-order Euler flight model."""
        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        gx, gy, gz = (float(self.physics.g[0]), float(self.physics.g[1]), float(self.physics.g[2]))
        k = self.physics.k
        dt_nominal = self.config.dt_integrate
        elapsed = 0.0
        while elapsed < duration - 1e-12:
            dt = min(dt_nominal, duration - elapsed)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            drag = -k * speed
            ax, ay, az = drag * vx + gx, drag * vy + gy, drag * vz + gz
            half_dt2 = 0.5 * dt * dt
            x += vx * dt + ax * half_dt2
            y += vy * dt + ay * half_dt2
            z += vz * dt + az * half_dt2
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            elapsed += dt
        return x, y, z, vx, vy, vz

    def _position_at_x(
        self, p0: np.ndarray, v0: np.ndarray, x_target: float, max_time: float,
    ) -> np.ndarray | None:
        """Return the interpolated flight position where x crosses ``x_target``."""
        if p0[0] >= x_target:
            return p0.copy()
        if v0[0] <= 0.0:
            return None

        dt_nominal = self.config.dt_integrate
        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        gx, gy, gz = (float(self.physics.g[0]), float(self.physics.g[1]), float(self.physics.g[2]))
        k = self.physics.k
        elapsed = 0.0
        while elapsed < max_time - 1e-12:
            dt = min(dt_nominal, max_time - elapsed)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            drag = -k * speed
            ax, ay, az = drag * vx + gx, drag * vy + gy, drag * vz + gz
            half_dt2 = 0.5 * dt * dt
            x_next = x + vx * dt + ax * half_dt2
            y_next = y + vy * dt + ay * half_dt2
            z_next = z + vz * dt + az * half_dt2
            if x <= x_target <= x_next:
                dx = x_next - x
                frac = (x_target - x) / dx if abs(dx) > 1e-9 else 0.0
                frac = min(1.0, max(0.0, frac))
                return np.array([
                    x + frac * (x_next - x),
                    y + frac * (y_next - y),
                    z + frac * (z_next - z),
                ])
            x, y, z = x_next, y_next, z_next
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            elapsed += dt
        return None

    def _face_opponent(self, n: np.ndarray) -> np.ndarray:
        """Orient a racket normal toward the opponent side (+x)."""
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        n = n / norm
        if n[0] < 0.0:
            n = -n
        if n[0] <= 1e-6:
            n = n + np.array([1e-6, 0.0, 0.0])
            n = n / np.linalg.norm(n)
        return n

    def _compute_racket_velocity(
        self, v_incoming: np.ndarray, v_outgoing: np.ndarray, C_r: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute desired racket velocity and face normal from impact model."""
        delta_v = v_outgoing - v_incoming
        delta_v_norm = np.linalg.norm(delta_v)

        if delta_v_norm < 1e-6:
            n = self._face_opponent(-v_incoming)
            return np.zeros(3), n

        u_hat = self._face_opponent(delta_v)
        v_o_n = np.dot(v_outgoing, u_hat)
        v_i_n = np.dot(v_incoming, u_hat)
        v_r_n = (v_o_n + C_r * v_i_n) / (1.0 + C_r)

        return v_r_n * u_hat, u_hat

    def _check_net_clearance(
        self, p_strike: np.ndarray, v_outgoing: np.ndarray, margin: float = 0.03,
    ) -> Tuple[bool, bool]:
        """Check height clearance and Y-axis net extent."""
        x_net = self.table.net_x
        z_net = self.table.net_height

        if v_outgoing[0] <= 0:
            return False, False

        p_net = self._position_at_x(
            p_strike, v_outgoing, x_net, max_time=self.config.max_predict_time
        )
        if p_net is None:
            return False, False

        z_at_net = p_net[2]
        y_at_net = p_net[1]

        y_net_min = -self.table.width - self.table.net_overhang
        y_net_max = self.table.net_overhang

        bypasses_posts = (y_at_net < y_net_min) or (y_at_net > y_net_max)
        if bypasses_posts:
            return False, True

        return z_at_net > (z_net + margin), False

    def plan(self, strike: StrikeTarget) -> RacketCommand:
        """Compute racket target state for a valid return."""
        if not strike.valid:
            return RacketCommand(
                p_intercept=strike.p_ball, v_racket=np.zeros(3),
                n_racket=np.array([1.0, 0.0, 0.0]), t_strike=strike.t_strike,
                v_ball_outgoing=np.zeros(3), target_land=self.config.target_land.copy(),
                clears_net=False, bypasses_net_posts=False, valid=False,
                num_bounces=strike.num_bounces,
            )

        p_strike = strike.p_ball
        v_incoming = strike.v_ball
        p_land = self.config.target_land.copy()

        v_outgoing = self._compute_outgoing_velocity(
            p_strike, p_land, self.config.delta_t_flight
        )
        v_racket, n_racket = self._compute_racket_velocity(
            v_incoming, v_outgoing, self.config.C_r
        )
        clears, bypasses = self._check_net_clearance(p_strike, v_outgoing)

        # Auto-adjust flight time if net clearance fails
        if not clears:
            for dt_adj in [0.4, 0.6, 0.35, 0.7, 0.3]:
                v_out_adj = self._compute_outgoing_velocity(p_strike, p_land, dt_adj)
                clears_adj, bypasses_adj = self._check_net_clearance(p_strike, v_out_adj)
                if clears_adj:
                    v_outgoing = v_out_adj
                    v_racket, n_racket = self._compute_racket_velocity(
                        v_incoming, v_outgoing, self.config.C_r
                    )
                    clears, bypasses = True, bypasses_adj
                    break

        return RacketCommand(
            p_intercept=p_strike, v_racket=v_racket, n_racket=n_racket,
            t_strike=strike.t_strike, v_ball_outgoing=v_outgoing,
            target_land=p_land, clears_net=clears,
            bypasses_net_posts=bypasses, valid=True,
            num_bounces=strike.num_bounces,
        )

"""Physical constants and tuning parameters for the HOPE planner.

Values follow the HOPE canonical world frame (origin at P1 near-side left
corner, X toward P2, Y left, Z up) and ITTF regulation table dimensions.
See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 2.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TableParams:
    """ITTF regulation table dimensions in the HOPE canonical frame."""

    length: float = 2.74          # m, along X
    width: float = 1.525          # m, along -Y
    height: float = 0.76          # m, table surface above floor
    net_x: float = 1.37           # m, net position along X
    net_height: float = 0.1525    # m, net height above table surface
    net_overhang: float = 0.15    # m, net extends past each table edge in Y


@dataclass
class BallPhysics:
    """Aerodynamic and restitution parameters.

    Calibrated from recorded ball trajectories by fitting observed
    accelerations and bounce velocity ratios (see calibrate_ball_physics).
    The HITTER paper uses 15 trajectories; more data improves robustness.
    """

    k: float = 0.5               # aerodynamic drag coefficient (s/m), 0.3-0.8 for 40 mm ball
    C_h: float = 0.75            # horizontal restitution coefficient
    C_v: float = 0.85            # vertical restitution coefficient (magnitude)
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    radius: float = 0.02         # ball radius, 40 mm diameter
    mass: float = 0.0027         # 2.7 g


@dataclass
class PlannerConfig:
    """Planner tuning parameters."""

    # State estimation
    poly_order: int = 2           # polynomial fit order
    fit_window: int =30 # 100         # number of position samples for velocity fit
    min_fit_samples: int = 10 # 6 # 40     # buffer must reach this many frames before velocity is computed
    mocap_hz: float = 180 # 360.0       # motion capture frame rate

    # Buffer-clear conditions (in addition to bounce detection). The estimation
    # buffer is reset whenever the ball is flying in the +x direction or has
    # travelled past clear_x_max, so the polynomial fit never spans a return leg.
    clear_x_max: float = 2.2      # reset buffer once ball x exceeds this (m)

    # Trajectory prediction
    dt_integrate: float = 0.001   # integration time step (s)
    max_predict_time: float = 2.0  # max forward prediction horizon (s)
    bounce_z_tol: float = 0.02 # 0.005   # z threshold for bounce detection (m)
    time_to_strike_invalid_x: float = 1.0  # invalidate TTS while inbound and ball x is beyond this

    # Racket planning
    x_hit: float = 0.0            # virtual hitting plane X coordinate (m)
    target_land: np.ndarray = field(
        default_factory=lambda: np.array([2.055, -0.7625, 0.0])
    )                             # center of opponent's half
    delta_t_flight: float = 0.5   # desired post-strike flight time (s)
    C_r: float = 0.88             # ball-racket coefficient of restitution
    racket_radius: float = 0.075  # 7.5 cm paddle radius
    # Stage 3 may use a coarser flight integration than Stage 2.  Defaults
    # retain the original common 1 ms model until explicitly overridden.
    stage3_dt_integrate: float = 0.001
    outgoing_velocity_tolerance_m: float = 1e-4

    # Publication workspace for the predicted interception point, in the
    # canonical HOPE world frame. A command outside this closed box is
    # suppressed before it reaches a ROS publisher.
    publish_intercept_x_min: float = 0.0
    publish_intercept_x_max: float = 0.0
    publish_intercept_y_min: float = 0.0
    publish_intercept_y_max: float = 1.5
    publish_intercept_z_min: float = 0.2
    publish_intercept_z_max: float = 0.6

    def allows_publish_intercept(self, point: np.ndarray) -> bool:
        """Return whether ``point`` is finite and inside the publish box.

        Bounds are inclusive: a target exactly on a configured plane remains
        valid. This is intentionally separate from physical trajectory
        validity; it is the final reachability gate for command publication.
        """
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return False
        return (
            self.publish_intercept_x_min <= p[0] <= self.publish_intercept_x_max
            and self.publish_intercept_y_min <= p[1] <= self.publish_intercept_y_max
            and self.publish_intercept_z_min <= p[2] <= self.publish_intercept_z_max
        )

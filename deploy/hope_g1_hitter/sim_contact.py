"""G1 1v1 paddle response and Magnus force for the shared analytical ball.

Uses the shared simulation physics profile and existing swept disk geometry. The contact
velocity includes racket angular velocity at the contact point, as in 1v1.
"""

import numpy as np
import torch
from hope_planner_g1.physics.ball_contact import orient_normal, predict_paddle_contact
from hope_planner_g1.profiles import load_profile


class G1Contact:
  def __init__(self, radius):
    profile = load_profile("sim")
    self.physics, self.config = profile.physics, profile.config
    if not np.isclose(radius, self.physics.radius):
      raise ValueError("Simulated ball radius differs from 1v1 physics profile")
    self.spin = np.zeros(3)
    self.state = None
    self.diagnostic = {}

  def add_magnus(self, position, velocity, incoming, dt):
    acceleration = self.config.k_m * np.cross(self.spin, incoming.numpy()[0])
    acceleration = torch.as_tensor(acceleration, dtype=incoming.dtype)[None]
    return position + 0.5 * acceleration * dt**2, velocity + acceleration * dt

  def response(self, incoming, center_velocity, normal, ball_center, racket_center):
    vin, n = incoming.numpy()[0], normal.numpy()[0]
    vr = getattr(self.state, "racket_velocity", None)
    vr = center_velocity.numpy()[0] if vr is None else np.asarray(vr)
    angular = getattr(self.state, "racket_angular_velocity", None)
    n = orient_normal(n, vin, vr)
    contact_point = ball_center.numpy()[0] - self.physics.radius * n
    if angular is not None:
      vr = vr + np.cross(angular, contact_point - racket_center.numpy()[0])
    outgoing, self.spin = predict_paddle_contact(
      vin, vr, n, self.spin, self.physics, self.config
    )
    self.diagnostic = dict(
      contact_model="g1_1v1",
      incoming_velocity=vin.tolist(),
      racket_velocity=vr.tolist(),
      racket_normal=n.tolist(),
      outgoing_spin=self.spin.tolist(),
    )
    return torch.as_tensor(outgoing, dtype=incoming.dtype)[None]

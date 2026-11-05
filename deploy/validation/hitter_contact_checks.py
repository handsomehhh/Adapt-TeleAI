"""Validate fitted collision response without altering legacy task physics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from deploy.backends.mujoco import SimBall
from deploy.hope_g1.world import mujoco_ball_config, simulation_translation
from deploy.hope_g1_hitter.sim_contact import G1Contact
from hope_planner_g1.physics.ball_contact import predict_paddle_contact
from hope_planner_g1.physics.constants import BallPhysics, PlannerConfig


def configuration():
  options = yaml.safe_load(
    Path("deploy/hope_g1_hitter/configs/rally_mujoco.yaml").read_text()
  )
  return mujoco_ball_config(options["ball"], simulation_translation(options["world"]))


@pytest.mark.parametrize("fitted", [False, True])
def test_simball_uses_selected_collision_response(fitted):
  cfg = configuration()
  if not fitted:
    cfg.pop("racket_contact_model")
  dt = 0.001
  ball = SimBall.from_launch(cfg, [0.023, 0, 1.005], [-4.0, 0.3, -0.5])
  state = SimpleNamespace(
    racket_pos=np.array([0.0, 0, 1.0]), racket_normal=np.array([1.0, 0, 0])
  )
  ball.substep(state, dt)
  assert ball.hit
  if fitted:
    contact = ball.contact
    expected, spin = predict_paddle_contact(
      np.asarray(contact["incoming_velocity"]),
      np.asarray(contact["racket_velocity"]),
      np.asarray(contact["racket_normal"]),
      np.zeros(3),
      BallPhysics(),
      PlannerConfig(),
    )
    np.testing.assert_allclose(ball.velocity, expected, atol=1e-6)
    np.testing.assert_allclose(ball.g1_contact.spin, spin, atol=1e-9)
    assert contact["contact_model"] == "g1_1v1"
  else:
    incoming = np.array([-4.0, 0.3, -0.5])
    incoming += dt * (
      -cfg["drag"] * np.linalg.norm(incoming) * incoming + [0, 0, -cfg["gravity"]]
    )
    expected = incoming.copy()
    expected[0] *= -cfg["racket_restitution"]
    np.testing.assert_allclose(ball.velocity, expected, atol=1e-6)
    assert not hasattr(ball, "g1_contact")


def test_contact_uses_angular_velocity_at_the_contact_point():
  law = G1Contact(0.02)
  law.state = SimpleNamespace(
    racket_velocity=np.array([1.0, 0.2, 0.3]),
    racket_angular_velocity=np.array([0.0, 3.0, 4.0]),
  )

  def tensor(x):
    return torch.tensor([x], dtype=torch.float64)

  response = law.response(
    tensor([-4, 0, -1]),
    tensor([0, 0, 0]),
    tensor([1, 0, 0]),
    tensor([0.02, 0.04, 0.01]),
    tensor([0, 0, 0]),
  )
  velocity = law.state.racket_velocity + np.cross(
    law.state.racket_angular_velocity, [0, 0.04, 0.01]
  )
  expected, spin = predict_paddle_contact(
    np.array([-4.0, 0, -1]),
    velocity,
    np.array([1.0, 0, 0]),
    np.zeros(3),
    law.physics,
    law.config,
  )
  np.testing.assert_allclose(response.numpy()[0], expected, atol=1e-9)
  np.testing.assert_allclose(law.spin, spin, atol=1e-8)


def test_generated_spin_changes_flight_with_1v1_magnus_coefficient():
  law = G1Contact(0.02)
  law.spin = np.array([0.0, 20.0, 0.0])
  position = torch.tensor([[0.0, 0, 1.0]], dtype=torch.float64)
  velocity = torch.tensor([[4.0, 0, 0.0]], dtype=torch.float64)
  p, v = law.add_magnus(position, velocity, velocity, 0.001)
  acceleration = law.config.k_m * np.cross(law.spin, velocity.numpy()[0])
  np.testing.assert_allclose(
    (v - velocity).numpy()[0], acceleration * 0.001, atol=1e-12
  )
  np.testing.assert_allclose(
    (p - position).numpy()[0], acceleration * 0.001**2 / 2, atol=1e-12
  )

"""Checks for hierarchy semantics, real FK, ball events and estimation."""

import inspect
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
  get_spec,
)
from mjlab.tasks.g1_adapt.actions import decode_speed
from mjlab.tasks.g1_adapt.ball import (
  BallPhysics,
  PositionEstimator,
  legal_feed_bank,
  predict_trajectory,
  swept_paddle,
)
from mjlab.tasks.g1_adapt.data import load_sequences, sample_windows
from mjlab.tasks.g1_adapt.generator import MotionGenerator
from mjlab.tasks.g1_adapt.motion import (
  G1Kinematics,
  blend_state,
  pack_state,
  quaternion_6d,
  rotation_6d,
)
from mjlab.tasks.g1_adapt.mvae import MotionVAE, MVAEConfig, load_mvae, save_mvae
from mjlab.tasks.g1_adapt.runner import AdaPTRallyRunner
from mjlab.tasks.g1_adapt.train_mvae import classification_loss


def initial_state(batch=3):
  root = torch.zeros(batch, 4)
  root[:, 0] = 1
  phase = torch.zeros(batch, 2)
  phase[:, 0] = 1
  return pack_state(
    torch.zeros(batch, 29),
    torch.zeros(batch, 29),
    root,
    torch.zeros(batch, 3),
    torch.zeros(batch, 3),
    phase,
  )


def test_batched_fk_matches_native_mujoco_for_random_joint_configurations():
  model = get_spec().compile()
  data = mujoco.MjData(model)
  fk = G1Kinematics()
  torch.manual_seed(8)
  q = torch.rand(5, 29) * 0.6 - 0.3
  roots = torch.tensor([[0.1, -0.2, 0.82]]).expand(5, -1)
  quat = torch.randn(5, 4)
  quat /= quat.norm(dim=-1, keepdim=True)
  positions, rotations = fk(roots, quat, q)
  for i in range(5):
    data.qpos[:3], data.qpos[3:7] = roots[i].numpy(), quat[i].numpy()
    for column, name in enumerate(G1_PINGPONG_JOINT_NAMES):
      data.qpos[model.joint(name).qposadr[0]] = q[i, column]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(positions[i].numpy(), data.xpos[1:], atol=3e-6)
    error = np.abs((rotations[i].numpy() * data.xquat[1:]).sum(-1))
    np.testing.assert_allclose(error, 1, atol=3e-6)


def test_rotation_roundtrip_including_half_turn():
  q = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.5, 0.5, 0.5, 0.5]])
  restored = quaternion_6d(rotation_6d(q))
  torch.testing.assert_close((restored * q).sum(-1).abs(), torch.ones(3))


def test_mvae_has_posterior_mixture_decoder_and_persisted_normalization(tmp_path):
  model = MotionVAE(MVAEConfig(hidden_dim=32, experts=3))
  history = initial_state()[:, None]
  target = initial_state() + 0.01
  prediction, stroke, spin, kl = model(history, target)
  loss = (
    prediction.square().mean()
    + classification_loss(stroke, torch.tensor([0, 1, -1]))
    + kl
  )
  loss.backward()
  assert model.posterior.weight.grad is not None
  assert model.layers[0].weight.grad is not None
  assert classification_loss(spin, torch.full((3,), -1)).item() == 0
  path = tmp_path / "mvae.pt"
  save_mvae(path, model, {"test": True})
  restored, metadata = load_mvae(path)
  assert metadata["test"]
  torch.testing.assert_close(model.mean, restored.mean)
  assert restored.decode(history, torch.zeros(3, 32))[0].shape == (3, 72)


def test_speed_adapted_state_is_used_for_next_autoregressive_step():
  model = MotionVAE(MVAEConfig(hidden_dim=32, experts=2))
  generator = MotionGenerator(model, 3, "cpu")
  previous = initial_state()
  generator.reset(
    torch.arange(3), previous, torch.tensor([[0.0, 0, 0.82]]).expand(3, -1)
  )
  z = torch.zeros(3, 32)
  alpha = torch.tensor([0.5, 1.0, 2.0])
  with torch.no_grad():
    proposal = model.decode(generator.history, z)[0]
  expected = blend_state(previous, proposal, alpha)
  expected[:, 12:41] = expected[:, 12:41].clamp(
    generator.fk.limits[:, 0], generator.fk.limits[:, 1]
  )
  expected[:, :3] = expected[:, :3].clamp(-3, 3)
  actual = generator.step(z, alpha)
  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(generator.history[:, -1], expected)
  torch.testing.assert_close(generator.last_alpha, alpha)


def test_multistep_autoregression_preserves_gradients_through_rotation_and_phase():
  model = MotionVAE(MVAEConfig(hidden_dim=32, experts=2))
  history = initial_state()[:, None]
  loss = torch.zeros(())
  for _ in range(4):
    proposed, _, _, kl = model(history, initial_state())
    state = blend_state(history[:, -1], proposed, torch.ones(3))
    loss = loss + state.square().mean() + kl
    history = state[:, None]
  loss.backward()
  gradients = [p.grad for p in model.layers[0].parameters()]
  assert all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_default_feeder_produces_full_training_bank():
  assert legal_feed_bank(BallPhysics(), seed=42).shape == (1024, 6)


def test_moving_paddle_collision_and_finite_disk():
  physics = BallPhysics(drag=0)
  old_ball = torch.tensor([[0.15, 0, 1.0], [0.15, 0.3, 1.0]])
  ball = old_ball.clone()
  ball[:, 0] = -0.05
  old_paddle = torch.tensor([[0.0, 0, 1.0]]).expand(2, -1)
  paddle = old_paddle.clone()
  paddle[:, 0] = 0.02
  normal = torch.tensor([[1.0, 0, 0]]).expand(2, -1)
  velocity = torch.tensor([[-4.0, 0, 0]]).expand(2, -1)
  hit, _, outgoing = swept_paddle(
    old_ball, ball, velocity, old_paddle, paddle, normal, 0.05, physics
  )
  assert hit.tolist() == [True, False]
  assert outgoing[0, 0] > 0
  torch.testing.assert_close(outgoing[1], velocity[1])


def test_predictor_includes_table_bounce():
  physics = BallPhysics(drag=0)
  trajectory = predict_trajectory(
    torch.tensor([[1.0, 0, 0.9]]), torch.tensor([[0.0, 0, -2.0]]), physics, 10, 0.01
  )
  assert trajectory[0, :, 2].min() >= physics.table_height + physics.radius
  assert trajectory[0, -1, 2] > trajectory[0, 5, 2]


def test_estimator_uses_only_received_positions_and_timestamps():
  estimator = PositionEstimator(1, "cpu")
  estimator.reset(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1))
  for i in range(1, 6):
    t = torch.tensor([i * 0.02])
    position = torch.tensor([[2 * t.item(), -t.item(), 1 - 0.5 * 9.81 * t.item() ** 2]])
    estimator.observe(position, t, torch.tensor([True]))
  position, velocity, valid, age = estimator.estimate(torch.tensor([0.1]))
  torch.testing.assert_close(
    velocity[0], torch.tensor([2.0, -1.0, -0.981]), atol=2e-5, rtol=1e-4
  )
  assert valid.item() and age.item() == 0
  assert position.isfinite().all()


def test_estimator_discards_duplicate_and_out_of_order_packets():
  estimator = PositionEstimator(1, "cpu")
  for t in (0.04, 0.04, 0.02):
    estimator.observe(torch.ones(1, 3), torch.tensor([t]), torch.tensor([True]))
  assert estimator.count.item() == 1
  assert not estimator.estimate(torch.tensor([0.06]))[2].item()


def test_feeder_is_reproducible_and_has_no_motion_conditioning():
  physics = BallPhysics()
  a = legal_feed_bank(physics, 16, 4)
  b = legal_feed_bank(physics, 16, 4)
  np.testing.assert_array_equal(a, b)
  assert a.shape == (16, 6)
  assert not any(
    "motion" in name for name in inspect.signature(legal_feed_bank).parameters
  )


def test_checkpoint_dependency_mismatch_fails_before_loading_policy(tmp_path):
  path = tmp_path / "planner.pt"
  torch.save({"infos": {"g1_adapt_dependencies": {"tracker": "wrong"}}}, path)
  runner = object.__new__(AdaPTRallyRunner)
  runner.dependencies = {"tracker": "right"}
  with pytest.raises(ValueError, match="dependencies"):
    runner.load(path)


def test_windows_do_not_cross_clip_boundaries_and_missing_spin_is_explicit():
  directory = Path(__file__).parents[1] / "dataset/g1_stage1/motions"
  if not directory.exists():
    pytest.skip("Optional motion archive not installed")
  clips = load_sequences(directory)
  windows, strokes, spins = sample_windows(clips, 5, 1, 4, "cpu", False)
  assert windows.shape == (5, 5, 72)
  assert (strokes >= 0).all()
  assert (spins == -1).all()


def test_speed_range_matches_stage1_training():
  torch.testing.assert_close(
    decode_speed(torch.tensor([-1.0, 0.0, 1.0])), torch.tensor([0.5, 1.0, 2.0])
  )

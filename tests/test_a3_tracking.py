"""CPU-only contracts for A3's ordered tracker and residual controller."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mjlab.asset_zoo.robots.agibot_a3.a3_constants import A3_JOINT_NAMES
from mjlab.tasks.a3_pingpong.actions import (
  FrozenTrackerActionCfg,
  OrderedJointPositionActionCfg,
)
from mjlab.tasks.a3_pingpong.config import a3_tracking_env_cfg, tracker_observations
from mjlab.tasks.a3_pingpong.tracking import (
  A3MotionCommand,
  A3MotionCommandCfg,
  joint_action_history,
)
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommandCfg
from mjlab.tasks.adapt_tennis.replay_joint_names import G1_REPLAY_JOINT_NAMES_27


def test_tracker_and_motion_use_same_explicit_joint_order():
  cfg = a3_tracking_env_cfg()
  command = cfg.commands["motion"]
  action = cfg.actions["joint_pos"]
  assert isinstance(command, MotionCommandCfg)
  assert isinstance(action, OrderedJointPositionActionCfg)
  assert command.joint_names == action.actuator_names == A3_JOINT_NAMES
  assert len(A3_JOINT_NAMES) == 31
  assert command.body_names[0] == "pelvis_link"
  for name in ("joint_pos", "joint_vel"):
    selected = cfg.observations["actor"].terms[name].params["asset_cfg"]
    assert selected.joint_names == A3_JOINT_NAMES
    assert selected.preserve_order
  assert list(cfg.observations["actor"].terms) == list(tracker_observations())
  # Adding A3 must not change G1's implicit mapping for existing tasks.
  assert MotionCommandCfg.__dataclass_fields__["joint_names"].default == (
    G1_REPLAY_JOINT_NAMES_27
  )


@pytest.fixture
def frozen_action(tmp_path):
  num_envs = 3
  count = len(A3_JOINT_NAMES)
  # The native robot order deliberately differs from the policy order to
  # detect accidental index-based scale/default-position/encoder handling.
  robot_names = tuple(reversed(A3_JOINT_NAMES))
  default_pos = torch.arange(count, dtype=torch.float32).repeat(num_envs, 1) / 100
  bias = torch.arange(count, dtype=torch.float32).repeat(num_envs, 1) / 1000
  entity = SimpleNamespace(
    find_joints=lambda names, preserve_order: (
      [robot_names.index(name) for name in names],
      list(names),
    ),
    data=SimpleNamespace(default_joint_pos=default_pos, encoder_bias=bias),
    set_joint_position_target=Mock(),
  )
  env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": entity})
  motion = SimpleNamespace(
    cfg=SimpleNamespace(fixed_dt=0.02),
    last_dt=torch.tensor([-0.005, 0.0, 0.01]),
    pending_dt=torch.full((num_envs,), 0.02),
  )

  def set_pending_dt(value):
    motion.pending_dt.copy_(value)
    motion.last_dt.copy_(value - motion.cfg.fixed_dt)

  motion.set_pending_dt = set_pending_dt
  env.command_manager = SimpleNamespace(get_term=lambda name: motion)
  obs_seen = []

  def compute_group(name):
    assert name == "tracker"
    obs = torch.zeros(num_envs, 171)
    obs[:, -32:-1] = action.joint_action
    obs[:, -1] = motion.last_dt
    obs_seen.append(obs.clone())
    return obs

  env.observation_manager = SimpleNamespace(compute_group=compute_group)
  # A deterministic exported actor whose output is visibly conditioned on dt.
  tracker = torch.nn.Linear(171, count)
  with torch.no_grad():
    tracker.weight.zero_()
    tracker.weight[:, -1] = 1.0
    tracker.bias.copy_(torch.arange(count, dtype=torch.float32) / 10)
  path = tmp_path / "tracker.pt"
  torch.jit.trace(tracker, torch.zeros(num_envs, 171)).save(str(path))
  cfg = FrozenTrackerActionCfg(
    entity_name="robot",
    actuator_names=A3_JOINT_NAMES,
    scale={name: float(index + 1) for index, name in enumerate(A3_JOINT_NAMES)},
    tracker_file=str(path),
  )
  action = cfg.build(env)
  env.action_manager = SimpleNamespace(get_term=lambda name: action)
  return action, env, motion, obs_seen


def test_frozen_tracker_preserves_dt_reference_alignment(frozen_action):
  action, env, motion, obs_seen = frozen_action
  old_dt = motion.last_dt.clone()
  old_history = torch.full_like(action.joint_action, 0.125)
  action.joint_action.copy_(old_history)
  raw = torch.zeros(3, 32)
  raw[:, -1] = torch.tensor([-1.0, 0.0, 1.0])
  action.process_actions(raw)
  # The tracker was trained with the dt that produced its current reference,
  # not a future speed command for the reference's next advance.
  torch.testing.assert_close(obs_seen[0][:, -1], old_dt)
  torch.testing.assert_close(obs_seen[0][:, -32:-1], old_history)
  torch.testing.assert_close(action.speed, torch.tensor([0.5, 1.0, 2.0]))
  torch.testing.assert_close(motion.pending_dt, torch.tensor([0.01, 0.02, 0.04]))
  assert action.action_dim == 32
  assert action.joint_action.shape == (3, 31)
  assert joint_action_history(env).shape == (3, 31)
  assert not action.tracker.training
  assert all(not p.requires_grad for p in action.tracker.parameters())


def test_residual_joint_scale_offset_and_encoder_compensation(frozen_action):
  action, env, _, obs_seen = frozen_action
  raw = torch.full((3, 32), 8.0)  # Residual clipping must occur before scaling.
  raw[:, -1] = 0.0
  action.process_actions(raw)
  base = action.tracker(obs_seen[0])
  expected_joint_action = base + 0.25 * 2.0
  torch.testing.assert_close(action.joint_action, expected_joint_action)
  action.apply_actions()
  robot = env.scene["robot"]
  target = robot.set_joint_position_target.call_args.args[0]
  ids = robot.set_joint_position_target.call_args.kwargs["joint_ids"]
  torch.testing.assert_close(ids, torch.arange(30, -1, -1))
  expected = (
    expected_joint_action * torch.arange(1, 32)
    + robot.data.default_joint_pos[:, ids]
    - robot.data.encoder_bias[:, ids]
  )
  torch.testing.assert_close(target, expected)

  retained = action.joint_action[1].clone()
  action.reset(torch.tensor([0, 2]))
  assert torch.count_nonzero(action.raw_action[[0, 2]]) == 0
  assert torch.count_nonzero(action.joint_action[[0, 2]]) == 0
  torch.testing.assert_close(action.speed[[0, 2]], torch.ones(2))
  torch.testing.assert_close(action.joint_action[1], retained)


def test_random_dt_reward_uses_completed_reference_advance():
  command = object.__new__(A3MotionCommand)
  command.cfg = A3MotionCommandCfg(
    entity_name="robot",
    body_names=("pelvis_link",),
    anchor_body_name="pelvis_link",
    resampling_time_range=(1e9, 1e9),
    random_dt_training_enabled=True,
    use_reference_time_discount=True,
  )
  command._fixed_dt = 0.02
  command.last_dt = torch.tensor([-0.01, 0.0, 0.02])
  # The base command consumes pending_dt when it updates the reference.
  command._pending_dt = torch.full((3,), 0.02)
  torch.testing.assert_close(
    command.reward_timeline_scale(), torch.tensor([0.5, 1.0, 2.0])
  )
  command.cfg.random_dt_training_enabled = False
  command._pending_dt = torch.tensor([0.04, 0.02, 0.01])
  torch.testing.assert_close(
    command.reward_timeline_scale(), torch.tensor([2.0, 1.0, 0.5])
  )
  command.cfg.use_reference_time_discount = False
  torch.testing.assert_close(command.reward_timeline_scale(), torch.ones(3))

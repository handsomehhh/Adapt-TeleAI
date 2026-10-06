"""Variable reference speed must not amplify timeout value bootstraps."""

import math

import pytest
import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


@pytest.mark.parametrize("step_scale", [None, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_timeout_value_is_not_integrated_as_reward(step_scale, gamma):
  obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
  groups = {"actor": ["policy"], "critic": ["policy"]}
  actor = MLPModel(
    obs,
    groups,
    "actor",
    1,
    hidden_dims=[4],
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[4])
  storage = RolloutStorage("rl", 2, 1, obs, [1])
  ppo = PPO(
    actor,
    critic,
    storage,
    gamma=gamma,
    normalize_advantage_per_mini_batch=True,
  )
  ppo.act(obs)
  ppo.transition.values.fill_(100)
  extras = {"time_outs": torch.tensor([True, False])}
  if step_scale is not None:
    extras["gae_step_scale"] = torch.full((2, 1), step_scale)
  ppo.process_env_step(obs, torch.full((2,), 2.0), torch.ones(2), extras)
  ppo.compute_returns(obs)

  scale = step_scale if step_scale is not None else 1.0
  if scale == 1.0:
    reward_coefficient = 1.0
  elif gamma == 1.0:
    reward_coefficient = scale
  else:
    reward_coefficient = (gamma**scale - 1.0) / math.log(gamma)
  expected_reward = reward_coefficient * 2.0
  expected_timeout_value = gamma**scale * 100.0
  torch.testing.assert_close(
    storage.returns[0, :, 0],
    torch.tensor([expected_reward + expected_timeout_value, expected_reward]),
    atol=1e-4,
    rtol=1e-5,
  )
  # With no immediate reward, the target coefficient is gamma**scale <= 1,
  # even for a fast motion step. The old implementation multiplied this by
  # integral(gamma**x, 0..scale), nearly doubling it when scale=2.
  assert float(storage.returns[0, 0, 0] - expected_reward) <= 100.0001
  storage.clear()
  assert not storage.timeout_bootstrap.any()

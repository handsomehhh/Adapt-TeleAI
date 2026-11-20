"""Load a training checkpoint or an existing JIT policy directly."""

from pathlib import Path

import numpy as np
import torch
import yaml

from deploy.motion_20261119.reference.g1_pingpong.core import JitPolicy


class TrainingYamlLoader(yaml.SafeLoader):
  """Read serialized configuration tuples/enums without executing objects."""


TrainingYamlLoader.add_constructor(
  "tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node)
)
TrainingYamlLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/object/apply:",
  lambda loader, suffix, node: loader.construct_sequence(node),
)


def training_config(path):
  return yaml.load(Path(path).read_text(), Loader=TrainingYamlLoader)


def run_directory(path):
  path = Path(path).resolve()
  return path.parent.parent if path.parent.name == "jit" else path.parent


class LoadedPolicy(JitPolicy):
  def __init__(self, model, input_dim, output_dim, iteration):
    self.model = model
    self.input_dim, self.output_dim = input_dim, output_dim
    self.iteration = iteration
    self(np.zeros(input_dim, dtype=np.float32))


def load_policy(path, input_dim, output_dim, agent_config=None):
  """No export step: instantiate the project's actor and load its saved weights."""
  path = Path(path).resolve()
  suffix = path.stem.rsplit("_", 1)[-1]
  iteration = int(suffix) if suffix.isdigit() else 0
  try:
    model = torch.jit.load(str(path), map_location="cpu").eval()
  except RuntimeError:
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint.get("iter", iteration))
    agent_config = agent_config or run_directory(path) / "params/agent.yaml"
    actor_cfg = training_config(agent_config)["actor"]
    if actor_cfg["class_name"] != "MLPModel":
      raise ValueError(
        "This G1 runtime requires a feed-forward MLPModel actor"
      ) from None
    actor = MLPModel(
      TensorDict({"actor": torch.zeros(1, input_dim)}, batch_size=[1]),
      {"actor": ["actor"]},
      "actor",
      output_dim,
      hidden_dims=actor_cfg["hidden_dims"],
      activation=actor_cfg["activation"],
      obs_normalization=actor_cfg["obs_normalization"],
      distribution_cfg=actor_cfg.get("distribution_cfg"),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model = actor.eval().as_jit().eval()
  return LoadedPolicy(model, input_dim, output_dim, iteration)

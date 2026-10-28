"""Actor-only CPU inference for the rsl_rl ``MLPModel`` hitter checkpoint."""

import numpy as np
import torch
from torch import nn


class HitterPolicy:
  input_dim = 104
  output_dim = 29

  def __init__(self, checkpoint, agent):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["actor_state_dict"]
    if any("normaliz" in key for key in state):
      raise ValueError("The hitter actor must not carry an observation normalizer")
    # distribution.std_param is exploration noise; deployment is deterministic.
    weights = {
      k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")
    }
    dims = [self.input_dim, *agent["actor"]["hidden_dims"], self.output_dim]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      layers.append(nn.Linear(a, b))
      if i < len(dims) - 2:
        layers.append(nn.ELU())
    self.actor = nn.Sequential(*layers).eval()
    self.actor.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(w).all() for w in self.actor.parameters()):
      raise ValueError("Invalid hitter actor tensors")
    self.iteration = int(saved["iter"])
    self(np.zeros(self.input_dim, dtype=np.float32))

  def __call__(self, observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.shape != (self.input_dim,) or not np.isfinite(x).all():
      raise ValueError("The hitter requires a finite 104-dimensional observation")
    with torch.inference_mode():
      action = self.actor(torch.from_numpy(x[None])).numpy()[0].copy()
    if action.shape != (self.output_dim,) or not np.isfinite(action).all():
      raise ValueError("Invalid hitter actor output")
    return action

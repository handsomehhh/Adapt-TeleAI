"""Conditional Motion VAE with a mixture-of-experts decoder.

Design references: Ling et al. Character Motion VAEs, Vid2Player3D, AdaPT.
Implemented for G1 tensors; no NVIDIA source code or pretrained models vendored.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)

from .motion import SCHEMA, STATE_DIM


def file_hash(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class MVAEConfig:
  state_dim: int = STATE_DIM
  latent_dim: int = 32
  hidden_dim: int = 256
  experts: int = 6
  history: int = 1
  fps: float = 50.0


class ExpertLinear(nn.Module):
  def __init__(self, inputs, outputs, experts):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(experts, outputs, inputs))
    self.bias = nn.Parameter(torch.zeros(experts, outputs))
    for weight in self.weight:
      nn.init.xavier_uniform_(weight)

  def forward(self, value, gates):
    values = torch.einsum("bi,eoi->beo", value, self.weight) + self.bias
    return torch.einsum("be,beo->bo", gates, values)


class MotionVAE(nn.Module):
  mean: torch.Tensor
  std: torch.Tensor

  def __init__(self, cfg: MVAEConfig):
    super().__init__()
    if cfg.state_dim != STATE_DIM or cfg.history < 1:
      raise ValueError("Unsupported motion schema/history")
    self.cfg = cfg
    size = cfg.state_dim * cfg.history
    self.register_buffer("mean", torch.zeros(cfg.state_dim))
    self.register_buffer("std", torch.ones(cfg.state_dim))
    self.encoder = nn.Sequential(
      nn.Linear(size + cfg.state_dim, cfg.hidden_dim),
      nn.ELU(),
      nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
      nn.ELU(),
    )
    self.posterior = nn.Linear(cfg.hidden_dim, 2 * cfg.latent_dim)
    self.gate = nn.Sequential(
      nn.Linear(size + cfg.latent_dim, 64), nn.ELU(), nn.Linear(64, cfg.experts)
    )
    self.layers = nn.ModuleList(
      [
        ExpertLinear(size + cfg.latent_dim, cfg.hidden_dim, cfg.experts),
        ExpertLinear(cfg.hidden_dim + cfg.latent_dim, cfg.hidden_dim, cfg.experts),
        ExpertLinear(cfg.hidden_dim + cfg.latent_dim, cfg.state_dim, cfg.experts),
      ]
    )
    self.stroke = nn.Linear(cfg.state_dim, 2)
    self.spin = nn.Linear(cfg.state_dim, 2)

  def normalize(self, value):
    return (value - self.mean) / self.std

  def decode(self, history, latent):
    condition = self.normalize(history).flatten(1)
    gates = self.gate(torch.cat((condition, latent), -1)).softmax(-1)
    hidden = condition
    for index, layer in enumerate(self.layers):
      hidden = layer(torch.cat((hidden, latent), -1), gates)
      if index < 2:
        hidden = F.elu(hidden)
    normalized = self.normalize(history[:, -1]) + hidden
    return (
      normalized * self.std + self.mean,
      self.stroke(normalized),
      self.spin(normalized),
    )

  def forward(self, history, target):
    context = self.normalize(history).flatten(1)
    encoded = self.encoder(torch.cat((context, self.normalize(target)), -1))
    mean, logvar = self.posterior(encoded).chunk(2, -1)
    logvar = logvar.clamp(-8, 4)
    latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
    state, stroke, spin = self.decode(history, latent)
    kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
    return state, stroke, spin, kl


def save_mvae(path, model, metadata, optimizer=None):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "schema": SCHEMA,
      "config": asdict(model.cfg),
      "model": model.state_dict(),
      "joint_names": G1_PINGPONG_JOINT_NAMES,
      "metadata": metadata,
      "optimizer": optimizer.state_dict() if optimizer is not None else None,
    },
    path,
  )


def load_mvae(path, device="cpu"):
  data = torch.load(Path(path), map_location=device, weights_only=False)
  if (
    data.get("schema") != SCHEMA
    or tuple(data.get("joint_names", ())) != G1_PINGPONG_JOINT_NAMES
  ):
    raise ValueError("MVAE motion schema or joint order mismatch")
  model = MotionVAE(MVAEConfig(**data["config"])).to(device)
  model.load_state_dict(data["model"], strict=True)
  if not all(torch.isfinite(p).all() for p in model.state_dict().values()):
    raise ValueError("Non-finite MVAE parameters/statistics")
  if (model.std <= 0).any():
    raise ValueError("Invalid MVAE normalization")
  return model, data["metadata"]

"""Train the G1 motion prior; split evaluation by whole clips, not frames."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from mjlab import TYRO_FLAGS

from .data import load_sequences, sample_windows
from .motion import blend_state
from .mvae import MotionVAE, MVAEConfig, save_mvae


@dataclass
class TrainMVAE:
  motion_directory: str = "dataset/g1_adapt_corrected/motions"
  output: str = "artifacts/g1_adapt/mvae.pt"
  labels_file: str | None = None
  epochs: int = 100
  batch_size: int = 256
  batches_per_epoch: int = 20
  rollout: int = 8
  latent_dim: int = 32
  hidden_dim: int = 256
  experts: int = 6
  history: int = 1
  learning_rate: float = 1e-3
  kl_weight: float = 0.01
  auxiliary_weight: float = 0.1
  validation_fraction: float = 0.2
  require_corrected: bool = True
  seed: int = 42
  device: str = "cuda:0"
  save_interval: int = 10
  select_best: bool = True
  velocity_consistency_weight: float = 0.0


def validation_windows(clips, history, rollout, device):
  """Fixed clip-local windows make validation scores comparable across epochs."""
  size = history + rollout
  windows = [
    clip["state"].unfold(0, size, 4).transpose(1, 2)
    for clip in clips
    if len(clip["state"]) >= size
  ]
  if not windows:
    raise ValueError("No held-out clip long enough for the requested rollout")
  return torch.cat(windows).to(device)


def classification_loss(logits, labels):
  known = labels >= 0
  return (
    F.cross_entropy(logits[known], labels[known]) if known.any() else logits.sum() * 0
  )


def train(cfg: TrainMVAE):
  if (
    cfg.epochs < 1
    or cfg.batches_per_epoch < 1
    or cfg.rollout < 1
    or cfg.save_interval < 1
  ):
    raise ValueError("epochs, batches and rollout must be positive")
  if not 0 < cfg.validation_fraction < 1:
    raise ValueError("Use a nonzero clip-level validation split")
  torch.manual_seed(cfg.seed)
  clips = load_sequences(cfg.motion_directory, cfg.labels_file)
  if len(clips) < 2:
    raise ValueError("At least two clips required for a held-out clip split")
  if cfg.require_corrected and not all(clip["corrected"] for clip in clips):
    raise ValueError(
      "Run correct_motions first, or explicitly set --require-corrected False for an uncorrected-data experiment"
    )
  order = torch.randperm(len(clips)).tolist()
  nval = max(1, min(len(clips) - 1, math.ceil(len(clips) * cfg.validation_fraction)))
  validation = [clips[i] for i in order[:nval]]
  training = [clips[i] for i in order[nval:]]
  fixed_validation = validation_windows(
    validation, cfg.history, cfg.rollout, cfg.device
  )
  model = MotionVAE(
    MVAEConfig(
      latent_dim=cfg.latent_dim,
      hidden_dim=cfg.hidden_dim,
      experts=cfg.experts,
      history=cfg.history,
    )
  ).to(cfg.device)
  states = torch.cat([clip["state"] for clip in training]).to(cfg.device)
  model.mean.copy_(states.mean(0))
  model.std.copy_(states.std(0).clamp_min(0.05))
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
  metadata = {
    "seed": cfg.seed,
    "physics_corrected": all(c["corrected"] for c in clips),
    "train_clips": [{k: c[k] for k in ("path", "sha256")} for c in training],
    "validation_clips": [{k: c[k] for k in ("path", "sha256")} for c in validation],
    "known_stroke_clips": sum(c["stroke"] >= 0 for c in clips),
    "known_spin_clips": sum(c["spin"] >= 0 for c in clips),
    "history": [],
    "training_configuration": asdict(cfg),
    "validation_windows": len(fixed_validation),
  }
  output = Path(cfg.output)
  checkpoints = output.parent / "checkpoints"
  checkpoints.mkdir(parents=True, exist_ok=True)
  best_metric, best_epoch = math.inf, 0
  for epoch in range(cfg.epochs):
    model.train()
    probability = max(0.0, 1 - epoch / max(1, cfg.epochs - 1))
    losses = []
    for _ in range(cfg.batches_per_epoch):
      sequence, stroke, spin = sample_windows(
        training, cfg.batch_size, cfg.history, cfg.rollout, cfg.device
      )
      history = sequence[:, : cfg.history]
      loss = torch.zeros((), device=cfg.device)
      for step in range(cfg.rollout):
        target = sequence[:, cfg.history + step]
        predicted, stroke_logits, spin_logits, kl = model(history, target)
        reconstruction = F.mse_loss(model.normalize(predicted), model.normalize(target))
        loss = (
          loss
          + reconstruction
          + cfg.kl_weight * kl
          + cfg.auxiliary_weight
          * (
            classification_loss(stroke_logits, stroke)
            + classification_loss(spin_logits, spin)
          )
        )
        prediction = blend_state(
          history[:, -1], predicted, torch.ones(cfg.batch_size, device=cfg.device)
        )
        if cfg.velocity_consistency_weight:
          finite_difference = (
            prediction[:, 12:41] - history[:, -1, 12:41]
          ) * model.cfg.fps
          mismatch = (finite_difference - prediction[:, 41:70]) / model.std[41:70]
          loss = loss + cfg.velocity_consistency_weight * mismatch.square().mean()
        teacher = torch.rand(cfg.batch_size, 1, device=cfg.device) < probability
        next_state = torch.where(teacher, target, prediction)
        history = torch.cat((history[:, 1:], next_state[:, None]), 1)
      loss = loss / cfg.rollout
      if not torch.isfinite(loss):
        raise RuntimeError("Non-finite MVAE objective")
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
      sequence = fixed_validation
      count = len(sequence)
      history = sequence[:, : cfg.history]
      # Prior-only autoregression is the deployment distribution; posterior reconstruction alone is insufficient.
      validation_loss = torch.zeros((), device=cfg.device)
      for step in range(cfg.rollout):
        predicted, _, _ = model.decode(
          history, torch.zeros(count, cfg.latent_dim, device=cfg.device)
        )
        predicted = blend_state(
          history[:, -1], predicted, torch.ones(count, device=cfg.device)
        )
        validation_loss += F.mse_loss(
          model.normalize(predicted), model.normalize(sequence[:, cfg.history + step])
        )
        history = torch.cat((history[:, 1:], predicted[:, None]), 1)
    record = {
      "epoch": epoch + 1,
      "loss": sum(losses) / len(losses),
      "held_out_prior_rollout_mse": float(validation_loss / cfg.rollout),
      "teacher_probability": probability,
    }
    metadata["history"].append(record)
    metadata["epochs"] = epoch + 1
    print(json.dumps(record), flush=True)
    save_mvae(cfg.output, model, metadata, optimizer)
    save_mvae(output.parent / "last.pt", model, metadata, optimizer)
    if record["held_out_prior_rollout_mse"] < best_metric:
      best_metric, best_epoch = record["held_out_prior_rollout_mse"], epoch + 1
      save_mvae(output.parent / "best.pt", model, metadata, optimizer)
    if (epoch + 1) % cfg.save_interval == 0 or epoch + 1 == cfg.epochs:
      save_mvae(checkpoints / f"epoch_{epoch + 1:04d}.pt", model, metadata, optimizer)
  metadata.update(
    selected_epoch=best_epoch if cfg.select_best else cfg.epochs,
    best_validation_mse=best_metric,
  )
  if cfg.select_best:
    selected = torch.load(
      output.parent / "best.pt", map_location="cpu", weights_only=False
    )
    selected["metadata"] = metadata
    torch.save(selected, output)
    model.load_state_dict(selected["model"])
  Path(cfg.output).with_suffix(".json").write_text(
    json.dumps(metadata, indent=2) + "\n"
  )
  return model, metadata


def main():
  train(tyro.cli(TrainMVAE, config=TYRO_FLAGS))


if __name__ == "__main__":
  main()

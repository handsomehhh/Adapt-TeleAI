"""Export matching rally/standing planner, MVAE and tracker dependencies."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mjlab import TYRO_FLAGS

from .data import load_clip
from .mvae import file_hash, load_mvae


@dataclass
class Export:
  rally_checkpoint: str
  standing_checkpoint: str
  ready_motion: str
  output: str = "artifacts/g1_adapt/bundle"


def export(cfg):
  output = Path(cfg.output)
  output.mkdir(parents=True, exist_ok=True)
  dependencies = None
  contract = None
  checkpoints = {}
  for role, filename in (
    ("rally", cfg.rally_checkpoint),
    ("standing", cfg.standing_checkpoint),
  ):
    path = Path(filename).resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = (data.get("infos") or {}).get("g1_adapt_dependencies")
    if not expected or (dependencies is not None and expected != dependencies):
      raise ValueError(
        "Rally and standing checkpoints require identical frozen dependencies"
      )
    dependencies = expected
    actual_contract = (data.get("infos") or {}).get("g1_adapt_contract")
    if not actual_contract or (contract is not None and actual_contract != contract):
      raise ValueError(
        "Rally and standing checkpoints require identical inference contracts"
      )
    contract = actual_contract
    for key, digest in dependencies.items():
      source = path.parent / "dependencies" / f"{key}.pt"
      if file_hash(source) != digest:
        raise ValueError(f"Checkpoint dependency differs: {key}")
      shutil.copy2(source, output / f"{key}.pt")
    actor = path.parent / "jit" / f"planner_{data['iter']}.pt"
    if not actor.is_file():
      raise FileNotFoundError(f"Missing JIT paired with checkpoint: {actor}")
    shutil.copy2(actor, output / f"{role}.pt")
    checkpoints[role] = {"checkpoint": str(path), "sha256": file_hash(path)}
  model, _ = load_mvae(output / "mvae.pt")
  initial = load_clip(cfg.ready_motion, model.cfg.fps)["state"][0]
  torch.save(
    {"state": initial, "source_sha256": file_hash(cfg.ready_motion)},
    output / "initial.pt",
  )
  planner = torch.jit.load(str(output / "rally.pt"), map_location="cpu")
  # Architecture contract: Stage-1 prefix replacing 29 actions by z+alpha, then generation history and ball observations.
  dimension = contract["planner_observation_dim"]
  with torch.no_grad():
    action = planner(torch.zeros(1, dimension))
    standing = torch.jit.load(str(output / "standing.pt"))(torch.zeros(1, dimension))
  if action.shape != (1, model.cfg.latent_dim + 1) or standing.shape != action.shape:
    raise ValueError("Planner export shape mismatch")
  files = ("mvae.pt", "tracker.pt", "rally.pt", "standing.pt", "initial.pt")
  manifest = {
    "schema": "g1-adapt-bundle-v1",
    "hashes": {name: file_hash(output / name) for name in files},
    **contract,
    "checkpoints": checkpoints,
    "hardware_io_included": False,
  }
  (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main():
  export(tyro.cli(Export, config=TYRO_FLAGS))


if __name__ == "__main__":
  main()

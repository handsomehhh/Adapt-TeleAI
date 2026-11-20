"""Resolve each module independently and enforce planner dependency contracts."""

import json
from pathlib import Path

import torch

from deploy import ROOT
from mjlab.tasks.g1_adapt.mvae import file_hash


def tracker_jit(path):
  path = Path(path).resolve()
  if path.name.startswith("model_"):
    path = path.parent / "jit" / f"modeljit_{path.stem.split('_')[-1]}.pt"
  if not path.is_file():
    raise FileNotFoundError(path)
  return path


class Artifacts:
  def __init__(self, run=None, bundle=None, overrides=None):
    self.files = {}
    self.verified_planners = {}
    self.contract = None
    self.run = Path(run).resolve() if run else None
    self.bundle = Path(bundle).resolve() if bundle else None
    if (
      self.bundle
      and self.run is None
      and (self.bundle.parent / "status.json").is_file()
    ):
      self.run = self.bundle.parent
    self.motion = None
    self.provenance = {}
    if self.run is None and self.bundle is None:
      runs = []
      for path in (ROOT / "logs/g1_adapt_full").glob("*/status.json"):
        if json.loads(path.read_text()).get("status") == "completed":
          runs.append(path.parent)
      if not runs:
        raise FileNotFoundError("Specify --run or --bundle; no completed run found")
      self.run = sorted(runs)[-1]
    if self.run:
      status = json.loads((self.run / "status.json").read_text())
      self.provenance = json.loads((self.run / "provenance.json").read_text())
      motions = sorted((self.run / "corrected/motions").glob("*.npz"))
      if motions:
        self.motion = motions[0]
      for stage, experiment in (
        ("tracker", "g1_stage1"),
        ("rally", "g1_adapt_rally"),
        ("standing", "g1_adapt_standing"),
      ):
        key = "stage1" if stage == "tracker" else stage
        checkpoint = status.get("stages", {}).get(key, {}).get("checkpoint")
        if checkpoint is None:
          available = list(
            (self.run / "work/logs/rsl_rl" / experiment).glob("*/model_*.pt")
          )
          if available:
            checkpoint = max(
              available, key=lambda p: (p.parent.name, int(p.stem.split("_")[-1]))
            )
        if checkpoint:
          self.files[stage] = Path(checkpoint)
      if (self.run / "mvae/mvae.pt").is_file():
        self.files["mvae"] = self.run / "mvae/mvae.pt"
    if self.bundle:
      info = json.loads((self.bundle / "bundle.json").read_text())
      if info.get("schema") != "g1-adapt-bundle-v1":
        raise ValueError("Unknown bundle schema")
      for name, digest in info["hashes"].items():
        if file_hash(self.bundle / name) != digest:
          raise ValueError(f"Bundle hash mismatch: {name}")
      self.contract = info
      self.files.update(
        {
          role: self.bundle / f"{role}.pt"
          for role in ("tracker", "mvae", "rally", "standing")
        }
      )
      for role in ("rally", "standing"):
        self.verified_planners[role] = (
          str(self.files[role]),
          file_hash(self.files[role]),
        )
    for role, value in (overrides or {}).items():
      if value:
        self.files[role] = Path(value).resolve()

  def require(self, roles):
    previous_files, previous_contract = dict(self.files), self.contract
    previous_verified = dict(self.verified_planners)
    try:
      return self._require(roles)
    except (ValueError, FileNotFoundError, KeyError):
      self.files, self.contract = previous_files, previous_contract
      self.verified_planners = previous_verified
      raise

  def _require(self, roles):
    for role in roles:
      path = self.files.get(role)
      if path is None or not path.is_file():
        raise FileNotFoundError(
          f"Missing {role}: select a completed checkpoint/run or override its path"
        )
      if role == "tracker":
        self.files[role] = tracker_jit(path)
    for role in (r for r in roles if r in ("rally", "standing")):
      path = self.files[role]
      if path.name.startswith("model_"):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        info = saved["infos"]
        contract = info["g1_adapt_contract"]
        if self.contract is not None:
          for key, value in contract.items():
            if self.contract.get(key) != value:
              raise ValueError(f"Planner inference contract mismatch: {key}")
        self.contract = contract
        for dependency, digest in info["g1_adapt_dependencies"].items():
          pinned = path.parent / "dependencies" / f"{dependency}.pt"
          selected = self.files.get(dependency, pinned)
          if dependency == "tracker":
            selected = tracker_jit(selected)
          if file_hash(selected) != digest:
            raise ValueError(
              f"{role} requires its exact frozen {dependency}; selected model differs"
            )
          self.files[dependency] = selected
        jit = path.parent / "jit" / f"planner_{saved['iter']}.pt"
        if not jit.is_file():
          raise FileNotFoundError(jit)
        self.files[role] = jit
        self.verified_planners[role] = (str(jit), file_hash(jit))
      elif self.verified_planners.get(role) != (str(path), file_hash(path)):
        raise ValueError(
          "Planner JIT requires a verified bundle or its paired training checkpoint"
        )
    return self

  def manifest(self):
    return {
      "run": str(self.run) if self.run else None,
      "bundle": str(self.bundle) if self.bundle else None,
      "models": {
        key: {"path": str(path), "sha256": file_hash(path)}
        for key, path in self.files.items()
        if path.is_file()
      },
      "contract": self.contract,
    }

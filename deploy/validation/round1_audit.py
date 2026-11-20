"""Recheck saved weights with the real training environment on CPU only."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from deploy import ROOT
from deploy.artifacts import Artifacts
from deploy.control import Reference
from deploy.session import Session
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_adapt.ball import BallPhysics, RallyBall
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg
from mjlab.tasks.g1_adapt.runner import inference_contract


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=32)
  parser.add_argument("--backend", choices=("training", "native"), default="training")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(2)
  run = ROOT / "logs/g1_adapt_full/2026-11-02_18-30-26_extended27mm_gpu0_env4096"
  artifacts = Artifacts(run)
  artifacts.require(["rally", "standing"])
  assert artifacts.contract is not None
  records = []
  start = time.monotonic()
  if args.backend == "training":
    cfg = g1_adapt_env_cfg(play=True)
    cfg.seed, cfg.scene.num_envs, cfg.episode_length_s = 20261104, 8, 4.0
    cfg.commands["motion"].motion_directory = str(run / "corrected/motions")
    cfg.commands["motion"].mvae_file = str(artifacts.files["mvae"])
    cfg.actions["joint_pos"].tracker_file = str(artifacts.files["tracker"])
    cfg.commands["ball"].physics = BallPhysics(**artifacts.contract["ball_physics"])
    cfg.commands["ball"].noise, cfg.commands["ball"].dropout = 0.005, 0.05
    for name in ("min_speed", "max_speed", "latent_clip"):
      setattr(cfg.actions["joint_pos"], name, artifacts.contract[name])
    rally = torch.jit.load(str(artifacts.files["rally"])).eval()
    standing = torch.jit.load(str(artifacts.files["standing"])).eval()
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      assert inference_contract(env) == artifacts.contract
      obs, _ = env.reset()
      with torch.inference_mode():
        for step in range(args.episodes * 201):
          ball = env.command_manager.get_term("ball")
          assert isinstance(ball, RallyBall)
          action = torch.where(
            ball.observed_valid[:, None], rally(obs["actor"]), standing(obs["actor"])
          )
          obs, _, terminated, truncated, _ = env.step(action)
          fell = env.termination_manager.get_term("fallen")
          for index in (terminated | truncated).nonzero().flatten().tolist():
            record = {k: float(v[index]) for k, v in ball.last_result.items()}
            record["fallen"] = float(fell[index])
            records.append(record)
          if step % 100 == 0:
            print("training CPU", step, "steps;", len(records), "trials", flush=True)
          if len(records) >= args.episodes:
            break
    finally:
      env.close()
  else:
    session = Session(
      artifacts, artifacts.motion, "hierarchy", seed=20261104, scenario="independent"
    )
    paths = sorted((run / "corrected/motions").glob("*.npz"))
    rng = torch.Generator().manual_seed(20261104)
    for index in range(args.episodes):
      path = paths[int(torch.randint(len(paths), (), generator=rng))]
      session.controller.reference = Reference(path)
      session.motion = path.resolve()
      session.reset()
      for _ in range(201):
        session.step()
        if len(session.trials) > index:
          break
      if index % 8 == 0:
        print("native CPU", index + 1, "trials", flush=True)
    records = session.trials
  records = records[: args.episodes]
  if len(records) < args.episodes:
    raise RuntimeError("Audit incomplete")
  result = {
    "protocol": "independent single-return, corrected motion pool, noise .005/dropout .05",
    "backend": args.backend + "_cpu",
    "episodes": len(records),
    "wall_seconds": time.monotonic() - start,
    "rates": {
      key: sum(r[key] for r in records) / len(records)
      for key in ("hit", "net_clear", "success", "fallen")
    },
    "trials": records,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
  main()

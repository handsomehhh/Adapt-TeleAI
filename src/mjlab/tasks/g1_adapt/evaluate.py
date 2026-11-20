"""Seeded single-return evaluation, with an explicit fixed-speed ablation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnv

from .ball import BallPhysics, RallyBall
from .env_cfg import g1_adapt_env_cfg
from .mvae import file_hash
from .runner import inference_contract


@dataclass
class Evaluate:
  checkpoint: str
  output: str = "artifacts/g1_adapt/evaluation.json"
  episodes: int = 200
  num_envs: int = 16
  seed: int = 10001
  noise: float = 0.005
  dropout: float = 0.05
  fixed_speed: bool = False
  device: str = "cuda:0"


def evaluate(cfg):
  if cfg.episodes < 1 or cfg.num_envs < 1:
    raise ValueError("episodes and num_envs must be positive")
  path = Path(cfg.checkpoint).resolve()
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  dependencies = checkpoint["infos"]["g1_adapt_dependencies"]
  config = g1_adapt_env_cfg(play=True)
  config.seed = cfg.seed
  config.scene.num_envs = cfg.num_envs
  config.episode_length_s = 4.0
  for key, digest in dependencies.items():
    source = path.parent / "dependencies" / f"{key}.pt"
    if file_hash(source) != digest:
      raise ValueError(f"Frozen {key} differs from checkpoint")
    if key == "mvae":
      config.commands["motion"].mvae_file = str(source)
    else:
      config.actions["joint_pos"].tracker_file = str(source)
  config.commands["ball"].noise = cfg.noise
  config.commands["ball"].dropout = cfg.dropout
  contract = checkpoint["infos"]["g1_adapt_contract"]
  # Restore the exact trained action/physics contract, independently of defaults.
  config.commands["ball"].physics = BallPhysics(**contract["ball_physics"])
  config.commands["ball"].prediction_steps = contract["prediction_steps"]
  config.actions["joint_pos"].scale = dict(
    zip(contract["joint_names"], contract["action_scales"], strict=True)
  )
  for name in ("min_speed", "max_speed", "latent_clip"):
    setattr(config.actions["joint_pos"], name, contract[name])
  planner = torch.jit.load(
    str(path.parent / "jit" / f"planner_{checkpoint['iter']}.pt"),
    map_location=cfg.device,
  ).eval()
  env = ManagerBasedRlEnv(config, device=cfg.device)
  records = []
  try:
    if inference_contract(env) != contract:
      raise ValueError("Evaluation environment differs from trained inference contract")
    observations, _ = env.reset()
    ball = env.command_manager.get_term("ball")
    assert isinstance(ball, RallyBall)
    with torch.inference_mode():
      for _ in range(cfg.episodes * 201):
        action = planner(observations["actor"])
        if cfg.fixed_speed:
          action[:, -1] = 0  # decode_speed(0) == 1; all other outputs unchanged.
        observations, _, terminated, truncated, _ = env.step(action)
        for index in (terminated | truncated).nonzero().flatten().tolist():
          records.append(
            {name: float(value[index]) for name, value in ball.last_result.items()}
          )
        if len(records) >= cfg.episodes:
          break
    if len(records) < cfg.episodes:
      raise RuntimeError("Evaluation did not complete the requested episodes")
    records = records[: cfg.episodes]
    report = {
      "configuration": asdict(cfg),
      "checkpoint_sha256": file_hash(path),
      "dependencies": dependencies,
      "episodes": len(records),
      "means": {
        name: sum(r[name] for r in records) / len(records) for name in records[0]
      },
      "trials": records,
      "protocol": "Independent single-return episodes; robot resets between trials",
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episodes", "means", "protocol")}))
    return report
  finally:
    env.close()


def main():
  evaluate(tyro.cli(Evaluate, config=TYRO_FLAGS))


if __name__ == "__main__":
  main()

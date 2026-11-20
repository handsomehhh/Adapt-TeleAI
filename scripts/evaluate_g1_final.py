"""Audit final rally/fixed-speed/standing policies using a run's source snapshot."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_adapt.ball import BallPhysics, RallyBall
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg
from mjlab.tasks.g1_adapt.mvae import file_hash
from mjlab.tasks.g1_adapt.runner import inference_contract


def audit(run, name, episodes, fixed=False, hierarchy=False):
  status = json.loads((run / "status.json").read_text())
  checkpoint = Path(status["stages"][name]["checkpoint"])
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  contract = saved["infos"]["g1_adapt_contract"]
  standing = name == "standing"
  cfg = g1_adapt_env_cfg(play=True, standing=standing)
  cfg.seed = 20261104
  cfg.scene.num_envs = 16
  cfg.episode_length_s = 8.0 if standing else 4.0
  cfg.commands["motion"].motion_directory = str(run / "corrected/motions")
  for kind, digest in saved["infos"]["g1_adapt_dependencies"].items():
    path = checkpoint.parent / "dependencies" / f"{kind}.pt"
    if file_hash(path) != digest:
      raise ValueError(f"Changed dependency: {kind}")
    if kind == "tracker":
      cfg.actions["joint_pos"].tracker_file = str(path)
    else:
      cfg.commands["motion"].mvae_file = str(path)
  cfg.commands["ball"].physics = BallPhysics(**contract["ball_physics"])
  cfg.commands["ball"].prediction_steps = contract["prediction_steps"]
  cfg.commands["ball"].noise = 0.005
  cfg.commands["ball"].dropout = 0.05
  cfg.actions["joint_pos"].scale = dict(
    zip(contract["joint_names"], contract["action_scales"], strict=True)
  )
  for kind in ("min_speed", "max_speed", "latent_clip"):
    setattr(cfg.actions["joint_pos"], kind, contract[kind])
  planner = torch.jit.load(
    str(checkpoint.parent / "jit" / f"planner_{saved['iter']}.pt"),
    map_location="cuda:0",
  ).eval()
  standing_planner = None
  if hierarchy:
    standing_checkpoint = Path(status["stages"]["standing"]["checkpoint"])
    standing_data = torch.load(
      standing_checkpoint, map_location="cpu", weights_only=False
    )
    if (
      standing_data["infos"]["g1_adapt_dependencies"]
      != saved["infos"]["g1_adapt_dependencies"]
      or standing_data["infos"]["g1_adapt_contract"] != contract
    ):
      raise ValueError(
        "Rally/standing policies have different frozen dependencies/contracts"
      )
    standing_planner = torch.jit.load(
      str(standing_checkpoint.parent / "jit" / f"planner_{standing_data['iter']}.pt"),
      map_location="cuda:0",
    ).eval()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  records = []
  try:
    if inference_contract(env) != contract:
      raise ValueError("Final audit inference contract differs from trained policy")
    obs, _ = env.reset()
    duration = torch.zeros(env.num_envs, device=env.device)
    with torch.inference_mode():
      for _ in range(episodes * (401 if standing else 201)):
        action = planner(obs["actor"])
        if standing_planner is not None:
          ball = env.command_manager.get_term("ball")
          assert isinstance(ball, RallyBall)
          action = torch.where(
            ball.observed_valid[:, None], action, standing_planner(obs["actor"])
          )
        if fixed:
          action[:, -1] = 0
        obs, _, terminated, truncated, _ = env.step(action)
        duration += env.step_dt
        # Read the stored termination predicate, not robot state after auto-reset.
        fell = env.termination_manager.get_term("fallen")
        ball = env.command_manager.get_term("ball")
        assert isinstance(ball, RallyBall)
        for index in (terminated | truncated).nonzero().flatten().tolist():
          record = {key: float(value[index]) for key, value in ball.last_result.items()}
          record["fallen"] = float(fell[index])
          record["episode_time_s"] = float(duration[index])
          records.append(record)
          duration[index] = 0
        if len(records) >= episodes:
          break
    if len(records) < episodes:
      raise RuntimeError("Final audit did not finish the requested trials")
    records = records[:episodes]
    return {
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": file_hash(checkpoint),
      "episodes": episodes,
      "seed": cfg.seed,
      "fixed_speed": fixed,
      "hierarchy_switching": hierarchy,
      "means": {key: sum(r[key] for r in records) / episodes for key in records[0]},
      "trials": records,
    }
  finally:
    env.close()
    torch.cuda.empty_cache()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=200)
  parser.add_argument("--standing-episodes", type=int, default=64)
  args = parser.parse_args()
  result: dict[str, Any] = {
    "protocol": "Independent single-return trials; standing 8s. Falls read from stored pre-reset termination predicates."
  }
  for key, stage, episodes, fixed, hierarchy in (
    ("rally", "rally", args.episodes, False, False),
    ("hierarchy", "rally", args.episodes, False, True),
    ("fixed_speed", "rally", args.episodes, True, True),
    ("standing", "standing", args.standing_episodes, False, False),
  ):
    result[key] = audit(args.run.resolve(), stage, episodes, fixed, hierarchy)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(key, json.dumps(result[key]["means"]), flush=True)
  result["completed"] = True
  args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()

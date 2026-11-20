"""Compare the complete native bridge to the real CPU training loop, one feed."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from deploy import ROOT
from deploy.artifacts import Artifacts
from deploy.session import Session
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_adapt.actions import LatentTrackerAction
from mjlab.tasks.g1_adapt.ball import BallPhysics, RallyBall
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  torch.set_num_threads(2)
  artifacts = Artifacts(
    ROOT / "logs/g1_adapt_full/2026-11-02_18-30-26_extended27mm_gpu0_env4096"
  )
  artifacts.require(["rally", "standing"])
  assert artifacts.contract is not None
  cfg = g1_adapt_env_cfg(play=True)
  cfg.seed, cfg.scene.num_envs, cfg.terminations = 42, 1, {}
  cfg.commands["motion"].motion_files = (str(artifacts.motion),)
  cfg.commands["motion"].motion_directory = ""
  cfg.commands["motion"].mvae_file = str(artifacts.files["mvae"])
  cfg.actions["joint_pos"].tracker_file = str(artifacts.files["tracker"])
  cfg.commands["ball"].physics = BallPhysics(**artifacts.contract["ball_physics"])
  cfg.commands["ball"].noise, cfg.commands["ball"].dropout = 0.0, 0.0
  cfg.commands["ball"].delay_steps = 0
  for name in ("min_speed", "max_speed", "latent_clip"):
    setattr(cfg.actions["joint_pos"], name, artifacts.contract[name])
  env = ManagerBasedRlEnv(cfg, device="cpu")
  rows = []
  try:
    obs, _ = env.reset()
    session = Session(
      artifacts,
      artifacts.motion,
      "hierarchy",
      scenario="independent",
      noise=0,
      dropout=0,
      delay_steps=0,
    )
    training_ball = env.command_manager.get_term("ball")
    assert isinstance(training_ball, RallyBall)
    session.ball.bank = torch.cat((training_ball.p, training_ball.v), -1)
    with torch.inference_mode():
      session.ball._resample_command(torch.tensor([0]))
    session.observe_ball(update=True)
    initial_error = float(
      np.abs(session.world.data.qpos - env.sim.data.qpos[0].numpy()).max()
    )
    with torch.inference_mode():
      for step in range(140):
        actor = obs["actor"]
        assert isinstance(actor, torch.Tensor)
        before = actor.clone()
        valid = bool(training_ball.observed_valid[0])
        action = (
          session.controller.planners["rally"]
          if valid
          else session.controller.planners["standing"]
        )(before)
        session.step()
        obs, _, _, _, _ = env.step(action)
        actual = session.controller.info["planner_observation"]
        diff = (actual - before).abs()
        action_term = env.action_manager.get_term("joint_pos")
        assert isinstance(action_term, LatentTrackerAction)
        row = {
          "step": step,
          "observation_error": float(diff.max()),
          "worst_index": int(diff.argmax()),
          "ball_position_error": float((session.ball.p - training_ball.p).abs().max()),
          "root_angular_velocity_error": float(
            (
              session.world.robot_state.root_angular_velocity_body
              - env.scene["robot"].data.root_link_ang_vel_b
            )
            .abs()
            .max()
          ),
          "target_error": float(
            (
              torch.tensor(session.records[-1]["targets"])[None]
              - action_term._processed_actions
            )
            .abs()
            .max()
          ),
          "qpos_error": float(
            np.abs(session.world.data.qpos - env.sim.data.qpos[0].numpy()).max()
          ),
        }
        rows.append(row)
        if step % 10 == 0:
          print(json.dumps(row), flush=True)
        if session.trials:
          break
    maxima = {
      key: max(row[key] for row in rows)
      for key in ("observation_error", "qpos_error", "target_error")
    }
    assert maxima["observation_error"] < 0.005, maxima
    assert maxima["qpos_error"] < 0.001, maxima
    assert maxima["target_error"] < 0.005, maxima
    result = {
      "passed": True,
      "initial_qpos_error": initial_error,
      "maxima": maxima,
      "rows": rows,
    }
    path = args.output or (
      ROOT / "deploy/recordings" / time.strftime("training_parity_%Y%m%d_%H%M%S.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
      "RESULT",
      json.dumps({"initial_qpos_error": initial_error, "final": rows[-1]}),
      flush=True,
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()

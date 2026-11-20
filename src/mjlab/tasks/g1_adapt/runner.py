"""Planner checkpoints pin frozen dependencies and export only the planner actor."""

import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from .mvae import file_hash


def inference_contract(env):
  raw = env.unwrapped
  cfg = raw.cfg
  action = raw.action_manager.get_term("joint_pos")
  robot = raw.scene["robot"]
  default = robot.data.default_joint_pos[0, action._target_ids]
  scale = torch.broadcast_to(
    torch.as_tensor(action._scale, device=default.device), (raw.num_envs, 29)
  )[0]
  return {
    "joint_names": list(G1_PINGPONG_JOINT_NAMES),
    "default_joints": default.cpu().tolist(),
    "action_scales": scale.cpu().tolist(),
    "control_dt": raw.step_dt,
    "prediction_steps": cfg.commands["ball"].prediction_steps,
    "min_speed": action.cfg.min_speed,
    "max_speed": action.cfg.max_speed,
    "latent_clip": action.cfg.latent_clip,
    "ball_physics": asdict(cfg.commands["ball"].physics),
    "planner_observation_dim": raw.observation_manager.group_obs_dim["actor"][0],
  }


class AdaPTRallyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
    super().__init__(env, train_cfg, log_dir, device)
    cfg = env.unwrapped.cfg
    sources = {
      "mvae": cfg.commands["motion"].mvae_file,
      "tracker": cfg.actions["joint_pos"].tracker_file,
    }
    self.dependencies = {key: file_hash(path) for key, path in sources.items()}
    self.contract = inference_contract(env)
    if log_dir:
      destination = Path(log_dir) / "dependencies"
      destination.mkdir(parents=True, exist_ok=True)
      for key, path in sources.items():
        target = destination / f"{key}.pt"
        if Path(path).resolve() != target.resolve():
          shutil.copy2(path, target)

  def save(self, path, infos=None):
    super().save(
      path,
      {
        **(infos or {}),
        "g1_adapt_dependencies": self.dependencies,
        "g1_adapt_contract": self.contract,
      },
    )
    destination = Path(path).parent / "jit"
    destination.mkdir(parents=True, exist_ok=True)
    torch.jit.script(self.alg.get_policy().as_jit().to("cpu").eval()).save(
      str(destination / f"planner_{self.current_learning_iteration}.pt")
    )

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = (data.get("infos") or {}).get("g1_adapt_dependencies")
    if expected != self.dependencies:
      raise ValueError(
        "Planner checkpoint was trained with different MVAE/tracker dependencies"
      )
    if (data.get("infos") or {}).get("g1_adapt_contract") != self.contract:
      raise ValueError(
        "Planner checkpoint inference contract differs from this environment"
      )
    return super().load(path, load_cfg, strict, map_location)

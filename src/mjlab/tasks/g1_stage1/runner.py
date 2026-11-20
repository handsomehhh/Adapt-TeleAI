"""G1 tracker checkpoints, adaptive sampling state and JIT export."""

from pathlib import Path

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner


class G1Stage1Runner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
    super().__init__(env, train_cfg, log_dir, device)

  def save(self, path, infos=None):
    motion = self.env.unwrapped.command_manager.get_term("motion")
    state = {"bin_failed_count": motion.bin_failed_count.detach().cpu()}
    super().save(path, {**(infos or {}), "g1_stage1_motion_state": state})
    destination = Path(path).parent / "jit"
    destination.mkdir(parents=True, exist_ok=True)
    policy = self.alg.get_policy().as_jit().to("cpu").eval()
    torch.jit.script(policy).save(
      str(destination / f"modeljit_{self.current_learning_iteration}.pt")
    )

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    state = (infos or {}).get("g1_stage1_motion_state", {})
    counts = state.get("bin_failed_count")
    motion = self.env.unwrapped.command_manager.get_term("motion")
    if counts is not None and counts.shape == motion.bin_failed_count.shape:
      motion.bin_failed_count.copy_(counts.to(motion.device))
    return infos

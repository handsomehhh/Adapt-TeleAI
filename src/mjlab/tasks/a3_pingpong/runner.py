"""Standard ping-pong outputs with reproducible frozen-tracker dependencies."""

import hashlib
import shutil
from pathlib import Path

import torch

from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand
from mjlab.tasks.adapt_tennis.rl import AdaPTTennisOnPolicyRunner


def tracker_sha256(path: str | Path) -> str:
  """Identify the actual frozen tracker bytes, independent of their location."""
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_tracker_dependency(expected: str | None, actual: str | None) -> None:
  """Legacy/Stage-1 checkpoints have no dependency; new Stage-2 ones must match."""
  if expected is not None and expected != actual:
    raise ValueError(
      "Stage-2 checkpoint frozen tracker mismatch: "
      f"expected SHA256 {expected}, loaded {actual}. "
      "Use the tracker.pt bundled with this checkpoint via --tracker-file "
      "(play) or --env.actions.joint-pos.tracker-file (train)."
    )


class PingPongOnPolicyRunner(AdaPTTennisOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
    super().__init__(env, train_cfg, log_dir, device, registry_name)
    action = env.unwrapped.action_manager.get_term("joint_pos")
    self._tracker_sha256 = (
      tracker_sha256(action.cfg.tracker_file)
      if hasattr(action.cfg, "tracker_file")
      else None
    )
    if log_dir and hasattr(action.cfg, "tracker_file"):
      # Stage-2 checkpoint remains portable even if the original run is moved.
      destination = Path(log_dir) / "tracker.pt"
      destination.parent.mkdir(parents=True, exist_ok=True)
      source = Path(action.cfg.tracker_file).expanduser().resolve()
      if source != destination.resolve():
        shutil.copy2(source, destination)

  def save(self, path, infos=None):
    motion = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(motion, MotionCommand)
    state = {"bin_failed_count": motion.bin_failed_count.detach().cpu()}
    checkpoint_infos = {**(infos or {}), "pingpong_motion_state_v1": state}
    if self._tracker_sha256 is not None:
      checkpoint_infos["pingpong_tracker_sha256"] = self._tracker_sha256
    super().save(path, checkpoint_infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    # Check before restoring actor, optimizer, or environment state. Loading a
    # same-shape but different tracker silently changes the hierarchical policy.
    metadata = (
      torch.load(path, map_location="cpu", weights_only=False).get("infos") or {}
    )
    expected_tracker = metadata.get(
      "pingpong_tracker_sha256", metadata.get("a3_tracker_sha256")
    )
    validate_tracker_dependency(expected_tracker, self._tracker_sha256)
    infos = super().load(path, load_cfg, strict, map_location)
    metadata = infos or {}
    state = metadata.get(
      "pingpong_motion_state_v1", metadata.get("a3_motion_state", {})
    )
    counts = state.get("bin_failed_count")
    motion = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(motion, MotionCommand)
    if counts is not None and counts.shape == motion.bin_failed_count.shape:
      motion.bin_failed_count.copy_(counts.to(motion.device))
    return infos


# Existing A3 configs and checkpoints retain a stable import path; load accepts
# both the old ``a3_*`` metadata keys and the robot-neutral versioned keys.
A3OnPolicyRunner = PingPongOnPolicyRunner

"""A receiving checkpoint must keep its original frozen low-level tracker."""

from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.a3_pingpong.runner import (
  A3OnPolicyRunner,
  PingPongOnPolicyRunner,
  tracker_sha256,
  validate_tracker_dependency,
)
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand
from mjlab.tasks.adapt_tennis.rl import AdaPTTennisOnPolicyRunner


def test_tracker_hash_tracks_content_not_path(tmp_path):
  original = tmp_path / "original.pt"
  moved = tmp_path / "tracker.pt"
  original.write_bytes(b"frozen tracker weights")
  moved.write_bytes(original.read_bytes())
  validate_tracker_dependency(tracker_sha256(original), tracker_sha256(moved))
  moved.write_bytes(b"different tracker weights")
  with pytest.raises(ValueError, match="tracker mismatch"):
    validate_tracker_dependency(tracker_sha256(original), tracker_sha256(moved))
  validate_tracker_dependency(None, tracker_sha256(moved))


def test_tracker_mismatch_rejected_before_parent_load(tmp_path, monkeypatch):
  checkpoint = tmp_path / "model.pt"
  torch.save({"infos": {"a3_tracker_sha256": "original"}}, checkpoint)
  runner = object.__new__(A3OnPolicyRunner)
  runner._tracker_sha256 = "different"
  parent_called = False

  def forbidden_load(*args, **kwargs):
    nonlocal parent_called
    parent_called = True
    raise AssertionError("Must validate before mutating model or optimizer state")

  monkeypatch.setattr(AdaPTTennisOnPolicyRunner, "load", forbidden_load)
  with pytest.raises(ValueError, match="expected SHA256 original, loaded different"):
    runner.load(str(checkpoint))
  assert not parent_called


def test_save_persists_tracker_hash_and_motion_sampler(monkeypatch):
  motion = object.__new__(MotionCommand)
  motion.bin_failed_count = torch.tensor([[1.0, 2.0]])
  runner = object.__new__(A3OnPolicyRunner)
  runner.env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: motion)
    )
  )
  runner._tracker_sha256 = "frozen"
  captured = {}

  def capture_save(self, path, infos):
    captured.update(infos)

  monkeypatch.setattr(AdaPTTennisOnPolicyRunner, "save", capture_save)
  runner.save("unused.pt", {"custom": 1})
  assert captured["pingpong_tracker_sha256"] == "frozen"
  assert captured["custom"] == 1
  torch.testing.assert_close(
    captured["pingpong_motion_state_v1"]["bin_failed_count"],
    motion.bin_failed_count,
  )


def test_a3_runner_name_remains_a_compatible_alias():
  assert A3OnPolicyRunner is PingPongOnPolicyRunner

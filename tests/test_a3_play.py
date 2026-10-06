"""Local task datasets must work without requiring a W&B motion artifact."""

from types import SimpleNamespace

from mjlab.scripts.play import PlayConfig, _has_configured_play_motion


def test_play_uses_local_task_dataset_and_honors_explicit_override(tmp_path):
  (tmp_path / "motion.npz").touch()
  motion = SimpleNamespace(
    motion_files=(), motion_file="", motion_directory=str(tmp_path)
  )
  assert _has_configured_play_motion(PlayConfig(), motion)
  # Invalid explicit arguments must fail rather than silently use a default.
  assert not _has_configured_play_motion(PlayConfig(motion_file="missing.npz"), motion)
  assert not _has_configured_play_motion(
    PlayConfig(registry_name="remote/motion"), motion
  )


def test_play_without_local_motion_preserves_artifact_fallback():
  motion = SimpleNamespace(motion_files=(), motion_file="", motion_directory="")
  assert not _has_configured_play_motion(PlayConfig(), motion)

"""Recovery must preserve completed training and continue the remaining stages."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_g1_adapt_full.py"
SPEC = importlib.util.spec_from_file_location("g1_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def recovery(tmp_path):
  root = tmp_path / "run"
  tracker = root / "work/tracker/model_29999.pt"
  snapshot_file = root / "snapshot/src/example.py"
  for path in (
    tracker,
    tracker.parent / "jit/modeljit_29999.pt",
    root / "mvae/mvae.pt",
    root / "corrected/motions/a.npz",
    root / "corrected/motions/b.npz",
    snapshot_file,
  ):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserved output")
  previous = dict(
    num_envs=4096,
    gpu=0,
    tracker_iterations=30000,
    mvae_epochs=100,
    rally_iterations=30000,
    standing_iterations=10000,
    save_interval=1000,
    evaluation_episodes=200,
  )
  state = dict(
    status="failed",
    stage="rally",
    pid=999999,
    error="out of memory",
    configuration=previous,
    stages={name: {"status": "completed"} for name in ("stage1", "correction", "mvae")},
  )
  state["stages"]["stage1"]["checkpoint"] = str(tracker)
  (root / "status.json").write_text(json.dumps(state))
  (root / "provenance.json").write_text(
    json.dumps({"source_files": {"src/example.py": MODULE.sha256(snapshot_file)}})
  )
  cfg = argparse.Namespace(resume_run=str(root), num_envs=1024, gpu=0)
  return root, cfg


def test_recovery_skips_completed_stages_and_preserves_budget(recovery, monkeypatch):
  root, cfg = recovery
  pipeline = MODULE.Pipeline(cfg)
  calls = []

  def train(name, task, experiment, iterations):
    calls.append((name, iterations))
    return root / f"{name}.pt"

  monkeypatch.setattr(pipeline, "train", train)
  monkeypatch.setattr(pipeline, "run", lambda name, *_: calls.append((name, None)))
  pipeline.execute()
  assert calls == [
    ("rally", 30000),
    ("standing", 10000),
    ("evaluate", None),
    ("evaluate_fixed_speed", None),
    ("export", None),
  ]
  assert cfg.num_envs == 1024
  assert pipeline.state["status"] == "completed"
  assert pipeline.state["restarts"][0]["previous_configuration"]["num_envs"] == 4096
  assert list(root.glob("status_before_*.json"))
  assert pipeline.env["ADAPT_G1_MOTION_DIR"] == str(root / "corrected/motions")
  assert pipeline.env["PYTHONPATH"].startswith(str(root / "snapshot/src"))


def test_recovery_rejects_changed_snapshot(recovery):
  root, cfg = recovery
  (root / "snapshot/src/example.py").write_text("changed")
  with pytest.raises(ValueError, match="snapshot changed"):
    MODULE.Pipeline(cfg)
  assert json.loads((root / "status.json").read_text())["status"] == "failed"


def test_recovery_refuses_to_discard_partial_planner(recovery):
  root, cfg = recovery
  checkpoint = root / "work/logs/rsl_rl/g1_adapt_rally/partial/model_1000.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.touch()
  with pytest.raises(ValueError, match="refusing to discard"):
    MODULE.Pipeline(cfg)


def test_checkpoint_discovery_ignores_failed_initialization(recovery, monkeypatch):
  root, cfg = recovery
  pipeline = MODULE.Pipeline(cfg)
  output = root / "work/logs/rsl_rl/g1_adapt_rally"
  (output / f"old_{root.name}_rally").mkdir(parents=True)
  completed = output / f"new_{root.name}_rally"
  completed.mkdir()
  (completed / "model_1000.pt").touch()
  final = completed / "model_29999.pt"
  final.touch()
  pipeline.state["stages"]["rally"] = {"status": "completed"}
  monkeypatch.setattr(pipeline, "run", lambda *_: None)
  assert pipeline.train("rally", "task", "g1_adapt_rally", 30000) == final


def test_resume_after_tracker_runs_remaining_stages_without_retraining(
  recovery, monkeypatch
):
  root, cfg = recovery
  state = json.loads((root / "status.json").read_text())
  state.update(status="interrupted", stage="stage1")
  del state["stages"]["correction"], state["stages"]["mvae"]
  (root / "status.json").write_text(json.dumps(state))
  (root / "mvae/mvae.pt").unlink()
  pipeline = MODULE.Pipeline(cfg)
  calls = []
  monkeypatch.setattr(pipeline, "correct", lambda *_: calls.append("correction"))
  monkeypatch.setattr(pipeline, "fit_mvae", lambda *_: calls.append("mvae"))
  monkeypatch.setattr(
    pipeline, "train_planners_and_export", lambda *_: calls.append("planners")
  )
  pipeline.execute()
  assert calls == ["correction", "mvae", "planners"]


def test_gpu_sharing_override_skips_foreign_process_wait(recovery, monkeypatch):
  _, cfg = recovery
  pipeline = MODULE.Pipeline(cfg)
  monkeypatch.setenv("ADAPT_G1_ALLOW_GPU_SHARING", "1")
  monkeypatch.setattr(
    MODULE.subprocess, "check_output", lambda *_args, **_kwargs: "GPU-test"
  )
  pipeline.wait_for_gpu()
  assert pipeline.state["gpu_sharing_allowed"] is True

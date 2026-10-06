"""CPU regression checks for receiving evaluation controls and outcomes."""

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/tools/evaluate_a3_pingpong.py"
)
spec = importlib.util.spec_from_file_location("evaluate_a3_pingpong", MODULE_PATH)
assert spec is not None and spec.loader is not None
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

ANALYZER_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/tools/analyze_a3_training.py"
)
analyzer_spec = importlib.util.spec_from_file_location(
  "analyze_a3_training", ANALYZER_PATH
)
assert analyzer_spec is not None and analyzer_spec.loader is not None
analyzer = importlib.util.module_from_spec(analyzer_spec)
analyzer_spec.loader.exec_module(analyzer)


@pytest.mark.parametrize(
  "fixed_speed,zero_residual",
  [(False, False), (True, False), (False, True), (True, True)],
)
def test_a3_receiving_ablation_keeps_other_action_components(
  fixed_speed, zero_residual
):
  actions = torch.arange(64, dtype=torch.float32).reshape(2, 32) + 0.25
  original = actions.clone()
  output = evaluate.ablate_receiving_actions(
    actions, fixed_speed=fixed_speed, zero_residual=zero_residual
  )
  torch.testing.assert_close(
    output[:, :-1],
    torch.zeros_like(actions[:, :-1]) if zero_residual else original[:, :-1],
  )
  torch.testing.assert_close(
    output[:, -1], torch.zeros_like(actions[:, -1]) if fixed_speed else original[:, -1]
  )
  torch.testing.assert_close(actions, original)


def test_a3_receiving_valid_landing_is_not_a_failure():
  # Successful return, missed return, fallen robot, feet failure, and timeout.
  reasons = {
    "ball_finished": torch.tensor([True, True, False, False, False]),
    "anchor_height": torch.tensor([False, False, True, False, False]),
    "anchor_orientation": torch.zeros(5, dtype=torch.bool),
    "feet_height": torch.tensor([False, False, False, True, False]),
    "time_out": torch.tensor([False, False, False, False, True]),
  }
  receiving = {"ball/state_missed": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])}
  failure, robot_failure = evaluate.receiving_failure_masks(reasons, receiving)
  torch.testing.assert_close(failure, torch.tensor([False, True, True, True, False]))
  torch.testing.assert_close(
    robot_failure, torch.tensor([False, False, True, True, False])
  )


def test_a3_landing_error_omits_zero_defaults_for_non_landings():
  receiving = {
    "ball/state_landing_error": torch.tensor([0.0, 0.1, 0.3, 0.0]),
    "ball/state_landed": torch.tensor([0.0, 1.0, 1.0, 0.0]),
    "ball/state_success": torch.tensor([0.0, 1.0, 0.0, 0.0]),
  }
  report = evaluate.receiving_landing_statistics(receiving)
  assert report["landing_error_mean_given_landing_m"] == pytest.approx(0.2)
  assert report["landing_error_mean_given_success_m"] == pytest.approx(0.1)
  assert report["landing_error_denominator_landings"] == 2
  assert report["landing_error_denominator_successes"] == 1
  receiving["ball/state_landed"].zero_()
  receiving["ball/state_success"].zero_()
  report = evaluate.receiving_landing_statistics(receiving)
  assert report["landing_error_mean_given_landing_m"] is None
  assert report["landing_error_mean_given_success_m"] is None


@pytest.mark.parametrize(
  "metadata_key", ["pingpong_tracker_sha256", "a3_tracker_sha256"]
)
def test_evaluator_validates_new_and_legacy_tracker_dependencies(
  tmp_path, metadata_key
):
  tracker = tmp_path / "tracker.pt"
  tracker.write_bytes(b"matching frozen tracker")
  checkpoint = tmp_path / "model_7.pt"
  torch.save(
    {"infos": {metadata_key: evaluate.sha256(tracker)}},
    checkpoint,
  )
  jit = tmp_path / "jit" / "modeljit_7.pt"
  jit.parent.mkdir()
  jit.write_bytes(b"exported policy")
  artifacts = evaluate.resolve_policy_artifacts(checkpoint)

  assert evaluate.validate_policy_tracker_dependency(
    artifacts, evaluate.sha256(tracker)
  ) == evaluate.sha256(tracker)
  with pytest.raises(ValueError, match="tracker mismatch"):
    evaluate.validate_policy_tracker_dependency(artifacts, "different")


def test_evaluator_run_directory_reports_resolved_checkpoint_sha(tmp_path):
  run = tmp_path / "run"
  (run / "jit").mkdir(parents=True)
  for index in (2, 10):
    torch.save({"infos": {}}, run / f"model_{index}.pt")
    (run / "jit" / f"modeljit_{index}.pt").write_bytes(f"jit {index}".encode())

  artifacts = evaluate.resolve_policy_artifacts(run)
  provenance = evaluate.policy_provenance(artifacts, None)
  expected_checkpoint = (run / "model_10.pt").resolve()
  assert artifacts.checkpoint == expected_checkpoint
  assert artifacts.jit == (run / "jit" / "modeljit_10.pt").resolve()
  assert provenance["resolved_checkpoint"] == str(expected_checkpoint)
  assert provenance["checkpoint_sha256"] == evaluate.sha256(expected_checkpoint)
  assert provenance["tracker_dependency_validated"] is False


def test_evaluator_standalone_jit_has_no_checkpoint_metadata(tmp_path):
  jit = tmp_path / "standalone.pt"
  jit.write_bytes(b"standalone jit")
  artifacts = evaluate.resolve_policy_artifacts(jit)
  provenance = evaluate.policy_provenance(artifacts, None)

  assert artifacts.checkpoint is None
  assert artifacts.jit == jit.resolve()
  assert evaluate.validate_policy_tracker_dependency(artifacts, "any tracker") is None
  assert provenance["checkpoint_sha256"] is None
  assert provenance["jit_sha256"] == evaluate.sha256(jit)
  assert provenance["tracker_dependency_validated"] is False


@pytest.mark.parametrize(
  "path_fragment,expected",
  [
    ("g1_pingpong_receive_stage2/run", True),
    ("a3_pingpong_receive_stage2/run", True),
    ("g1_pingpong_tracking_stage1_random_dt/run", False),
  ],
)
def test_analyzer_stage_detection_uses_robot_neutral_path_fallback(
  tmp_path, path_fragment, expected
):
  assert analyzer.is_receiving_run([], tmp_path / path_fragment) is expected


def test_analyzer_stage_detection_prefers_saved_command_config(tmp_path):
  run = tmp_path / "arbitrary_run_name"
  params = run / "params"
  params.mkdir(parents=True)
  (params / "env.yaml").write_text(
    "commands:\n  motion:\n    fixed_dt: 0.02\n  ball:\n    max_ball_time: 3.0\n"
  )
  assert analyzer.is_receiving_run([], run)
  assert analyzer.is_receiving_run(["Metrics/ball/hit_rate"], tmp_path / "unknown")

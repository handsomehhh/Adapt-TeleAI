"""Round-two decisions, G1 reward changes and preserved MVAE checkpoints."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.g1_adapt import train_mvae
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg
from mjlab.tasks.g1_adapt.mvae import load_mvae
from mjlab.tasks.g1_adapt.terms import task_reward
from mjlab.tasks.g1_stage1.env_cfg import g1_stage1_env_cfg
from mjlab.tasks.g1_stage1.mdp.tracking import racket_tracking

SPEC = importlib.util.spec_from_file_location(
  "g1_supervisor", Path(__file__).parents[1] / "scripts/supervise_g1_adapt.py"
)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


def evidence(
  success=0.0, falls=0.1, fixed_success=0.2, center=0.07, generation_falls=5
):
  audit = {"completed": True}
  for name, win in (
    ("rally", success),
    ("fixed_speed", fixed_success),
    ("standing", 0.0),
  ):
    audit[name] = {
      "means": {"success": win, "fallen": falls, "hit": 0.9, "net_clear": 0.8}
    }
  components = {
    "tracker_strike": [
      {
        "speed": 1.0,
        "at_nearest_strike_frame": {"racket_center_error_m": {"mean": center}},
      }
    ],
    "generated_tracking": {
      "summary": [
        {"condition": "zero", "speed": 1.0, "falls": generation_falls, "clips": 9}
      ]
    },
  }
  return audit, components


def test_decisions_change_only_when_evidence_crosses_thresholds():
  audit, components = evidence()
  decision = SUPERVISOR.select_adjustments(audit, components)
  assert decision["launch"]["tracker_iterations"] == 45000
  assert decision["launch"]["rally_iterations"] == 45000
  assert decision["profile"]["mvae"]["rollout"] == 32
  assert decision["profile"]["planner"]["max_speed"] == 1.5
  assert decision["profile"]["planner"]["latent_clip"] == 2.0
  audit, components = evidence(
    success=0.8, fixed_success=0.8, falls=0, center=0.02, generation_falls=0
  )
  decision = SUPERVISOR.select_adjustments(audit, components)
  assert decision["launch"]["tracker_iterations"] == 30000
  assert decision["launch"]["rally_iterations"] == 30000
  assert decision["profile"]["tracker"] == {}
  assert "max_speed" not in decision["profile"]["planner"]
  audit["completed"] = False
  with pytest.raises(ValueError, match="must complete"):
    SUPERVISOR.select_adjustments(audit, components)


def test_training_profile_preserves_dimensions_and_changes_only_requested_terms(
  tmp_path, monkeypatch
):
  decision = SUPERVISOR.select_adjustments(*evidence())
  path = tmp_path / "profile.json"
  path.write_text(json.dumps(decision["profile"]))
  monkeypatch.setenv("ADAPT_G1_TRAINING_PROFILE", str(path))
  tracker = g1_stage1_env_cfg()
  assert tracker.rewards["racket_center"].weight == 1.0
  assert tracker.rewards["racket_normal"].weight == 0.5
  assert len(tracker.commands["motion"].joint_names) == 29
  rally = g1_adapt_env_cfg()
  assert rally.actions["joint_pos"].latent_dim == 32
  assert rally.actions["joint_pos"].max_speed == 1.5
  assert rally.rewards["landing"].weight == 200
  assert rally.rewards["hit"].params["event_rewards"]
  monkeypatch.delenv("ADAPT_G1_TRAINING_PROFILE")
  assert "racket_center" not in g1_stage1_env_cfg().rewards
  assert g1_adapt_env_cfg().actions["joint_pos"].max_speed == 2.0


def test_racket_reward_uses_physical_center_and_normal():
  reference_position = torch.zeros(1, 1, 3)
  reference_quat = torch.tensor([[[1.0, 0, 0, 0]]])
  data = SimpleNamespace(
    body_link_pos_w=reference_position.clone(), body_link_quat_w=reference_quat.clone()
  )
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=["right_wrist_yaw_link"]),
    robot=SimpleNamespace(body_names=["right_wrist_yaw_link"], data=data),
    body_pos_w=reference_position,
    body_quat_w=reference_quat,
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    num_envs=1,
    device="cpu",
  )
  assert racket_tracking(env, 0.08).item() == 1.0
  assert racket_tracking(env, 0.25, normal=True).item() == 1.0
  data.body_link_pos_w[..., 0] = 0.1
  assert racket_tracking(env, 0.08).item() < 0.3
  assert racket_tracking(env, 0.25, normal=True).item() == 1.0


def test_event_rewards_do_not_pay_again_for_latched_hit_or_net():
  ball = SimpleNamespace(
    hit=torch.tensor([True]),
    hit_event=torch.tensor([False]),
    net_clear=torch.tensor([True]),
    net_event=torch.tensor([False]),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: ball), scene={"robot": None}
  )
  assert task_reward(env, "hit").item() == 1.0
  assert task_reward(env, "hit", event_rewards=True).item() == 0.0
  assert task_reward(env, "net", event_rewards=True).item() == 0.0
  ball.hit_event[:] = True
  assert task_reward(env, "hit", event_rewards=True).item() == 1.0


def test_mvae_saves_periodic_best_last_and_selects_best(tmp_path, monkeypatch):
  torch.set_num_threads(2)
  clips = []
  for index in range(3):
    state = torch.zeros(40, 72)
    state[:, 6] = state[:, 10] = 1
    angle = torch.arange(40) * 0.08 + index * 0.1
    state[:, 70], state[:, 71] = angle.cos(), angle.sin()
    state[:, 12:41] = angle.sin()[:, None] * 0.03
    state[:, 41:70] = angle.cos()[:, None] * 0.12
    clips.append(
      {
        "state": state,
        "corrected": True,
        "stroke": index % 2,
        "spin": -1,
        "path": str(index),
        "sha256": str(index),
      }
    )
  monkeypatch.setattr(train_mvae, "load_sequences", lambda *_: clips)
  output = tmp_path / "mvae.pt"
  cfg = train_mvae.TrainMVAE(
    output=str(output),
    epochs=3,
    batch_size=4,
    batches_per_epoch=2,
    rollout=32,
    hidden_dim=16,
    experts=2,
    save_interval=2,
    device="cpu",
    velocity_consistency_weight=0.01,
  )
  model, metadata = train_mvae.train(cfg)
  assert (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()
  assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
    "epoch_0002.pt",
    "epoch_0003.pt",
  ]
  best = min(metadata["history"], key=lambda v: v["held_out_prior_rollout_mse"])
  assert metadata["selected_epoch"] == best["epoch"]
  selected, _ = load_mvae(output)
  best_model, _ = load_mvae(tmp_path / "best.pt")
  for key, value in selected.state_dict().items():
    torch.testing.assert_close(value, best_model.state_dict()[key])
    torch.testing.assert_close(value, model.state_dict()[key])


def test_supervisor_waits_for_completion_then_audits_before_round_two(
  tmp_path, monkeypatch
):
  first = tmp_path / "first"
  first.mkdir()
  (first / "mvae").mkdir()
  (first / "mvae/mvae.pt").write_bytes(b"mvae")
  tracker = first / "tracker/model_29999.pt"
  tracker.parent.mkdir()
  (tracker.parent / "jit").mkdir()
  (tracker.parent / "jit/modeljit_29999.pt").write_bytes(b"tracker")
  state = {
    "status": "running",
    "pid": 999999,
    "stages": {"stage1": {"checkpoint": str(tracker)}},
  }
  SUPERVISOR.write(first / "status.json", state)
  audit, components = evidence()
  components.update(
    mvae_sha256=hashlib.sha256(b"mvae").hexdigest(),
    tracker_sha256=hashlib.sha256(b"tracker").hexdigest(),
  )
  component_path = tmp_path / "components.json"
  SUPERVISOR.write(component_path, components)
  supervisor = object.__new__(SUPERVISOR.Supervisor)
  supervisor.root = tmp_path
  supervisor.state = {}
  supervisor.cfg = argparse.Namespace(
    first_run=first, component_report=component_path, runs_root=tmp_path, gpu=0
  )
  calls = []

  def sleep(_):
    calls.append("wait")
    state["status"] = "completed"
    SUPERVISOR.write(first / "status.json", state)

  def run(name, script, args, source_run=None):
    calls.append(name)
    if name.startswith("audit"):
      SUPERVISOR.write(Path(args[args.index("--output") + 1]), audit)
    else:
      second = Path(args[args.index("--run-directory") + 1])
      second.mkdir()
      SUPERVISOR.write(second / "status.json", {"status": "completed"})

  monkeypatch.setattr(supervisor, "heartbeat", lambda: None)
  monkeypatch.setattr(supervisor, "run", run)
  monkeypatch.setattr(SUPERVISOR.time, "sleep", sleep)
  monkeypatch.setattr(SUPERVISOR.os, "kill", lambda *_: None)
  monkeypatch.setattr(SUPERVISOR, "shutil_disk_free", lambda _: 100 * 1024**3)
  supervisor.execute()
  assert calls == ["wait", "audit_first_final", "train_round2", "audit_second_final"]
  assert supervisor.state["status"] == "completed"
  assert (tmp_path / "decision.json").is_file()


def test_duplicate_supervisor_is_rejected_before_gpu_access(tmp_path):
  gpu = 99123
  lock = Path(f"/tmp/adapt-g1-gpu{gpu}.lock")
  try:
    with lock.open("a+") as held:
      fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
      with pytest.raises(RuntimeError, match="already holds"):
        SUPERVISOR.Supervisor(argparse.Namespace(campaign=tmp_path, gpu=gpu))
  finally:
    lock.unlink(missing_ok=True)

"""Watch round one, audit its final models, and run one adapted full second round.

Adaptation uses explicit metric thresholds, not a background language-model
session. GPU locks are advisory; external applications can bypass them.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]


def now():
  return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def read(path):
  return json.loads(Path(path).read_text())


def write(path, value):
  path = Path(path)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def select_adjustments(audit, components):
  """Bounded engineering choices from final policy metrics and component tests."""
  if not audit.get("completed"):
    raise ValueError("Final policy audit must complete before selecting round two")
  rally = audit.get("hierarchy", audit["rally"])["means"]
  fixed, standing = (audit[name]["means"] for name in ("fixed_speed", "standing"))
  for metrics in (rally, fixed, standing):
    for key in ("success", "fallen", "hit", "net_clear"):
      value = float(metrics[key])
      if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid final metric {key}: {value}")
  strike = next(row for row in components["tracker_strike"] if row["speed"] == 1.0)
  center_error = strike["at_nearest_strike_frame"]["racket_center_error_m"]["mean"]
  generation = next(
    row
    for row in components["generated_tracking"]["summary"]
    if row["speed"] == 1 and row["condition"] == "zero"
  )
  improve_tracker = center_error > 0.05
  improve_prior = generation["falls"] / generation["clips"] > 0.1
  poor_landing = rally["success"] < 0.5
  unstable = rally["fallen"] > 0.05 or standing["fallen"] > 0.05
  speed_harmful = (
    fixed["success"] > rally["success"] + 0.05
    or fixed["fallen"] + 0.05 < rally["fallen"]
  )
  profile: dict[str, Any] = {
    "schema": "g1-adapt-training-profile-v1",
    "tracker": {},
    "mvae": {
      "rollout": 32 if improve_prior else 16,
      "batch_size": 128,
      "batches_per_epoch": 40,
      "save_interval": 10,
      "select_best": True,
      "velocity_consistency_weight": 0.01 if improve_prior else 0.0,
    },
    "planner": {
      "event_rewards": True,
      "rally_weights": {
        "hit": 2.0,
        "net": 5.0,
        "landing": 200.0 if poor_landing else 100.0,
        "forward": 0.0 if poor_landing else 1.0,
        "tracking": 0.2,
        "fall": -50.0,
      },
      "standing_weights": {"fall": -50.0},
    },
  }
  reasons = [
    "Save periodic/last/best MVAE checkpoints; select best using fixed held-out windows.",
    "Reward hit/net transitions once per control step instead of repeatedly paying for a latched result.",
  ]
  if improve_tracker:
    profile["tracker"] = {
      "reward_weights": {"motion_wrist": 1.0, "motion_hit_arm": 1.0},
      "racket_center_weight": 1.0,
      "racket_center_std": 0.08,
      "racket_normal_weight": 0.5,
      "racket_normal_std": 0.25,
    }
    reasons.append(
      f"Strike center error {center_error:.4f}m exceeds 0.05m: add physical racket-center/normal tracking and increase tracker budget."
    )
  if improve_prior:
    reasons.append(
      "Unplanned 5-second generation falls exceed 10%: extend MVAE autoregression to 32 frames and add a small joint-velocity consistency objective."
    )
  if poor_landing:
    reasons.append(
      f"Final landing success {rally['success']:.3f} is below 0.5: emphasize landing, remove the forward-speed incentive and increase rally training."
    )
  if unstable:
    cast(dict[str, Any], profile["planner"])["latent_clip"] = 2.0
    profile["planner"]["rally_weights"]["tilt"] = -4.0
    profile["planner"]["standing_weights"]["tilt"] = -4.0
    reasons.append(
      "Final rally/standing fall rate exceeds 5%: restrict latent magnitude and strengthen balance penalties."
    )
  if speed_harmful:
    cast(dict[str, Any], profile["planner"]).update(min_speed=0.75, max_speed=1.5)
    reasons.append(
      "Fixed speed improves landing or falls by >5 percentage points: narrow speed range to 0.75--1.5."
    )
  return {
    "policy_version": 1,
    "created_at": now(),
    "reasons": reasons,
    "evidence": {
      "rally": rally,
      "fixed_speed": fixed,
      "standing": standing,
      "strike_center_error_m": center_error,
      "unplanned_generation": generation,
    },
    "profile": profile,
    "launch": {
      "tracker_num_envs": 4096,
      "planner_num_envs": 1024,
      "tracker_iterations": 45000 if improve_tracker else 30000,
      "mvae_epochs": 200 if improve_prior else 100,
      "rally_iterations": 45000 if poor_landing else 30000,
      "standing_iterations": 15000 if unstable else 10000,
      "save_interval": 1000,
      "evaluation_episodes": 200,
    },
  }


class Supervisor:
  def __init__(self, cfg):
    self.cfg = cfg
    self.root = cfg.campaign.resolve()
    self.child = None
    self.lock = Path(f"/tmp/adapt-g1-gpu{cfg.gpu}.lock").open("a+")
    try:
      fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      raise RuntimeError(
        "A G1 GPU supervisor/pipeline already holds this GPU lock"
      ) from None
    self.uuid = subprocess.check_output(
      ["nvidia-smi", "-i", str(cfg.gpu), "--query-gpu=uuid", "--format=csv,noheader"],
      text=True,
    ).strip()
    first = read(cfg.first_run / "status.json")
    self.first_group = int(first["pid"])
    self.state: dict[str, Any] = {
      "status": "running",
      "phase": "waiting_first_round",
      "pid": os.getpid(),
      "started_at": now(),
      "gpu": cfg.gpu,
      "gpu_uuid": self.uuid,
      "first_run": str(cfg.first_run.resolve()),
      "second_run": None,
      "exclusive_mode": "advisory_lock_and_process_monitor",
      "driver_exclusive_error": "Insufficient Permissions",
    }
    self.heartbeat()

  def gpu_processes(self):
    result = subprocess.check_output(
      [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
      ],
      text=True,
    )
    processes = []
    for line in result.splitlines():
      device, pid = (part.strip() for part in line.split(","))
      if device == self.uuid:
        try:
          processes.append({"pid": int(pid), "group": os.getpgid(int(pid))})
        except ProcessLookupError:
          continue
    return processes

  def heartbeat(self):
    processes = self.gpu_processes()
    allowed = {self.first_group, os.getpgrp()}
    if self.child is not None:
      allowed.add(self.child.pid)
    previous = self.state.get("foreign_gpu_pids", [])
    foreign = [p["pid"] for p in processes if p["group"] not in allowed]
    if foreign != previous:
      print(f"[{now()}] External GPU processes changed: {foreign}", flush=True)
      with (self.root / "events.jsonl").open("a") as events:
        events.write(json.dumps({"at": now(), "foreign_gpu_pids": foreign}) + "\n")
    self.state.update(
      updated_at=now(),
      gpu_processes=processes,
      foreign_gpu_pids=foreign,
      free_disk_gib=round(shutil_disk_free(self.root) / 1024**3, 2),
      gpu_sharing_allowed=os.environ.get("ADAPT_G1_ALLOW_GPU_SHARING") == "1",
    )
    write(self.root / "status.json", self.state)

  def phase(self, name):
    self.state["phase"] = name
    print(f"[{now()}] {name}", flush=True)
    self.heartbeat()

  def wait_gpu_idle(self):
    if os.environ.get("ADAPT_G1_ALLOW_GPU_SHARING") == "1":
      return
    while self.gpu_processes():
      self.state["waiting_for_gpu_idle"] = True
      self.heartbeat()
      time.sleep(15)
    self.state.pop("waiting_for_gpu_idle", None)

  def run(self, name, script, arguments, source_run=None):
    self.wait_gpu_idle()
    self.phase(name)
    env = dict(os.environ)
    env.update(
      CUDA_VISIBLE_DEVICES=str(self.cfg.gpu),
      PYTHONDONTWRITEBYTECODE="1",
      ADAPT_G1_GPU_LOCK_FD=str(self.lock.fileno()),
    )
    if source_run:
      snapshot = source_run / "snapshot"
      env["PYTHONPATH"] = f"{snapshot / 'src'}:{snapshot / 'src/rsl_rl_lib'}"
      env["ADAPT_G1_MOTION_DIR"] = str(source_run / "corrected/motions")
      env.pop("ADAPT_G1_TRAINING_PROFILE", None)
      if (snapshot / "training_profile.json").is_file():
        env["ADAPT_G1_TRAINING_PROFILE"] = str(snapshot / "training_profile.json")
    log_path = self.root / f"{name}.log"
    with log_path.open("x") as log:
      self.child = subprocess.Popen(
        [sys.executable, "-u", str(script), *map(str, arguments)],
        cwd=REPO,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(self.lock.fileno(),),
      )
      self.state.update(child_pid=self.child.pid, active_log=str(log_path))
      self.heartbeat()
      while self.child.poll() is None:
        try:
          self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
          if self.state.get("second_run"):
            path = Path(self.state["second_run"]) / "status.json"
            if path.is_file():
              second = read(path)
              self.state["second_progress"] = {
                key: second.get(key)
                for key in ("status", "stage", "updated_at", "stages")
              }
          self.heartbeat()
      code = self.child.returncode
    self.child = None
    self.state.pop("child_pid", None)
    if code:
      raise RuntimeError(f"{name} exited {code}; inspect {log_path}")
    self.heartbeat()

  def execute(self):
    continuation = getattr(self.cfg, "continue_second_run", None)
    if continuation:
      second = continuation.resolve()
      self.state["second_run"] = str(second)
      self.state["first_progress"] = {"status": "completed"}
      self.run(
        "continue_round2_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        REPO / "scripts/train_g1_adapt_full.py",
        ["--gpu", self.cfg.gpu, "--resume-run", second, "--exclusive-gpu"],
      )
      self.finish_second(second)
      return
    while True:
      first = read(self.cfg.first_run / "status.json")
      self.state["first_progress"] = {
        key: first.get(key) for key in ("status", "stage", "updated_at", "stages")
      }
      self.heartbeat()
      if first["status"] == "completed":
        break
      if first["status"] in ("failed", "interrupted"):
        raise RuntimeError(f"First round did not complete: {first.get('error')}")
      # Do not wait silently forever on a dead pipeline or stale status file.
      try:
        os.kill(int(first["pid"]), 0)
      except ProcessLookupError:
        raise RuntimeError(
          "First-round worker exited without a completed status"
        ) from None
      time.sleep(15)
    final = self.root / "first_final_audit.json"
    self.run(
      "audit_first_final",
      REPO / "scripts/evaluate_g1_final.py",
      ["--run", self.cfg.first_run, "--output", final],
      self.cfg.first_run,
    )
    components = read(self.cfg.component_report)
    checkpoint = Path(first["stages"]["stage1"]["checkpoint"])
    tracker = (
      checkpoint.parent / "jit" / f"modeljit_{checkpoint.stem.split('_')[-1]}.pt"
    )
    for name, path in (
      ("mvae", self.cfg.first_run / "mvae/mvae.pt"),
      ("tracker", tracker),
    ):
      if hashlib.sha256(path.read_bytes()).hexdigest() != components[f"{name}_sha256"]:
        raise ValueError("Component report does not match the final frozen models")
    decision = select_adjustments(read(final), components)
    write(self.root / "decision.json", decision)
    profile = self.root / "round2_profile.json"
    write(profile, decision["profile"])
    if shutil_disk_free(self.root) < 8 * 1024**3:
      raise RuntimeError(
        "Less than 8 GiB free disk; refusing to lose checkpoints mid-run"
      )
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    second = self.cfg.runs_root / f"{stamp}_extended27mm_gpu{self.cfg.gpu}_round2"
    self.state.update(
      second_run=str(second.resolve()), decision=str(self.root / "decision.json")
    )
    arguments = [
      "--gpu",
      self.cfg.gpu,
      "--exclusive-gpu",
      "--run-directory",
      second,
      "--training-profile",
      profile,
    ]
    for key, value in decision["launch"].items():
      arguments.extend(["--" + key.replace("_", "-"), value])
    self.run("train_round2", REPO / "scripts/train_g1_adapt_full.py", arguments)
    self.finish_second(second)

  def finish_second(self, second):
    if read(second / "status.json")["status"] != "completed":
      raise RuntimeError("Second-round worker exited without completing its pipeline")
    self.run(
      "audit_second_final",
      REPO / "scripts/evaluate_g1_final.py",
      ["--run", second, "--output", self.root / "second_final_audit.json"],
      second,
    )
    self.state.update(status="completed", phase="two_rounds_completed", ended_at=now())
    self.heartbeat()


def shutil_disk_free(path):
  stat = os.statvfs(path)
  return stat.f_bavail * stat.f_frsize


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--first-run", type=Path, required=True)
  parser.add_argument("--component-report", type=Path, required=True)
  parser.add_argument("--campaign", type=Path, required=True)
  parser.add_argument("--runs-root", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--continue-second-run", type=Path)
  cfg = parser.parse_args()
  supervisor = Supervisor(cfg)

  def stop(_signum, _frame):
    raise KeyboardInterrupt("Supervisor terminated")

  signal.signal(signal.SIGTERM, stop)
  try:
    supervisor.execute()
  except BaseException as error:
    if supervisor.child is not None and supervisor.child.poll() is None:
      os.killpg(supervisor.child.pid, signal.SIGTERM)
      supervisor.child.wait(timeout=60)
    supervisor.state.update(
      status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
      error=str(error),
      ended_at=now(),
    )
    supervisor.heartbeat()
    raise


if __name__ == "__main__":
  main()

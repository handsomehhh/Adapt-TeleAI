"""Run a sequential G1 AdaPT training pipeline on one physical GPU.

Source code, robot assets and raw motions are copied into the run directory.
Every child process uses that snapshot. A failed stage stops the pipeline.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]


def now():
  return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Pipeline:
  def __init__(self, cfg):
    self.cfg = cfg
    if cfg.resume_run:
      self.restore()
      return
    self.root = Path(cfg.run_directory).resolve()
    self.root.mkdir(parents=True, exist_ok=False)
    self.work = self.root / "work"
    self.work.mkdir()
    shutil.copy2(__file__, self.root / "pipeline.py")
    self.snapshot = self.root / "snapshot"
    self.child = None
    self.state: dict[str, Any] = {
      "status": "preparing",
      "pid": os.getpid(),
      "started_at": now(),
      "configuration": vars(cfg),
      "run_directory": str(self.root),
      "stages": {},
    }
    self.save()
    shutil.copytree(
      REPO / "src",
      self.snapshot / "src",
      ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    self.raw = self.snapshot / "dataset/g1_stage1/motions"
    shutil.copytree(REPO / "dataset/g1_stage1", self.raw.parent)
    profile_path = getattr(cfg, "training_profile", None)
    if profile_path:
      shutil.copy2(profile_path, self.snapshot / "training_profile.json")
    asset = self.snapshot / "src/mjlab/asset_zoo/robots/unitree_g1_pingpong"
    manifest = json.loads((asset / "asset_manifest.json").read_text())
    validation = json.loads((asset / "asset_validation.json").read_text())
    if sha256(asset / "g1.xml") != validation["mjcf_sha256"]:
      raise ValueError("Training robot differs from independently validated asset")
    source = asset / manifest["source"]["local_path"]
    if sha256(source) != validation["source_urdf_sha256"]:
      raise ValueError("Source URDF differs from validated source")
    if source.name != "g1_29dof_pingpong_fast_extended_27mm.urdf":
      raise ValueError("Full run requires the updated extended-27mm URDF")
    revision = REPO / "source_revision.json"
    git_head = (
      json.loads(revision.read_text())["git_head"]
      if revision.is_file()
      else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
      ).strip()
    )
    provenance = {
      "created_at": now(),
      "python": sys.executable,
      "git_head": git_head,
      "asset_manifest": manifest,
      "asset_validation": validation,
      "source_files": {
        str(p.relative_to(self.snapshot)): sha256(p)
        for p in sorted(self.snapshot.rglob("*"))
        if p.is_file()
      },
    }
    (self.root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    self.env = dict(os.environ)
    self.env.update(
      CUDA_VISIBLE_DEVICES=str(cfg.gpu),
      PYTHONDONTWRITEBYTECODE="1",
      PYTHONPATH=f"{self.snapshot / 'src'}:{self.snapshot / 'src/rsl_rl_lib'}",
      ADAPT_G1_MOTION_DIR=str(self.raw),
      OMP_NUM_THREADS="8",
      MKL_NUM_THREADS="8",
    )
    if profile_path:
      self.env["ADAPT_G1_TRAINING_PROFILE"] = str(
        self.snapshot / "training_profile.json"
      )
    self.state.update(
      status="running",
      snapshot=str(self.snapshot),
      source_urdf_sha256=manifest["source"]["sha256"],
    )
    self.save()

  def restore(self):
    """Reuse completed stages after Tracker and before any planner checkpoint."""
    self.root = Path(self.cfg.resume_run).resolve()
    self.work = self.root / "work"
    self.snapshot = self.root / "snapshot"
    self.raw = self.snapshot / "dataset/g1_stage1/motions"
    self.child = None
    self.state = json.loads((self.root / "status.json").read_text())
    if self.state["status"] not in ("failed", "interrupted"):
      raise ValueError("Only a failed/interrupted pipeline can be resumed")
    if self.state["stages"].get("stage1", {}).get("status") != "completed":
      raise ValueError("Resume requires completed stage1")
    if self.state.get("stage") not in ("stage1", "correction", "mvae", "rally"):
      raise ValueError("Resume supports completed Tracker before planner training")
    # A partially trained planner needs checkpoint resume and an adjusted budget.
    partial = list((self.work / "logs/rsl_rl/g1_adapt_rally").glob("*/model_*.pt"))
    if partial:
      raise ValueError("Rally checkpoints exist; refusing to discard planner progress")
    provenance = json.loads((self.root / "provenance.json").read_text())
    for relative, expected in provenance["source_files"].items():
      if sha256(self.snapshot / relative) != expected:
        raise ValueError(f"Training snapshot changed: {relative}")
    tracker = Path(self.state["stages"]["stage1"]["checkpoint"])
    tracker_jit = tracker.parent / "jit" / f"modeljit_{tracker.stem.split('_')[-1]}.pt"
    mvae = self.root / "mvae/mvae.pt"
    required = [tracker, tracker_jit]
    if self.state["stages"].get("mvae", {}).get("status") == "completed":
      required.append(mvae)
    for path in required:
      if not path.is_file():
        raise FileNotFoundError(path)
    if (
      self.state["stages"].get("correction", {}).get("status") == "completed"
      and len(list((self.root / "corrected/motions").glob("*.npz"))) < 2
    ):
      raise ValueError("Corrected motion dataset is missing")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    shutil.copy2(self.root / "status.json", self.root / f"status_before_{stamp}.json")
    shutil.copy2(__file__, self.root / f"pipeline_resume_{stamp}.py")
    self.state.setdefault("restarts", []).append(
      {
        "resumed_at": now(),
        "previous_configuration": self.state["configuration"],
        "previous_error": self.state.get("error"),
        "tracker_sha256": sha256(tracker_jit),
        "mvae_sha256": sha256(mvae) if mvae.is_file() else None,
      }
    )
    # Preserve all original training budgets; only GPU/environment count may change.
    previous = self.state["configuration"]
    for name in (
      "tracker_iterations",
      "mvae_epochs",
      "rally_iterations",
      "standing_iterations",
      "save_interval",
      "evaluation_episodes",
    ):
      setattr(self.cfg, name, previous[name])
    for name in ("tracker_num_envs", "planner_num_envs"):
      if getattr(self.cfg, name, None) is None and previous.get(name):
        setattr(self.cfg, name, previous[name])
    self.cfg.run_directory = str(self.root)
    self.env = dict(os.environ)
    self.env.update(
      CUDA_VISIBLE_DEVICES=str(self.cfg.gpu),
      PYTHONDONTWRITEBYTECODE="1",
      PYTHONPATH=f"{self.snapshot / 'src'}:{self.snapshot / 'src/rsl_rl_lib'}",
      ADAPT_G1_MOTION_DIR=str(self.raw),
      OMP_NUM_THREADS="8",
      MKL_NUM_THREADS="8",
    )
    self.state.update(
      status="running", pid=os.getpid(), configuration=vars(self.cfg), resumed_at=now()
    )
    if (self.snapshot / "training_profile.json").is_file():
      self.env["ADAPT_G1_TRAINING_PROFILE"] = str(
        self.snapshot / "training_profile.json"
      )
    for key in ("error", "ended_at", "child_pid"):
      self.state.pop(key, None)
    self.save()

  def save(self):
    self.state["updated_at"] = now()
    path = self.root / "status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(self.state, indent=2) + "\n")
    temporary.replace(path)

  def run(self, name, module, arguments):
    if getattr(self.cfg, "exclusive_gpu", False):
      self.wait_for_gpu()
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    log_path = self.root / f"{name}.log"
    if log_path.exists():
      stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
      log_path.rename(self.root / f"{name}_before_{stamp}.log")
    stage = {
      "status": "running",
      "started_at": now(),
      "command": command,
      "log": str(log_path),
    }
    self.state["stage"] = name
    self.state["stages"][name] = stage
    print(f"[{now()}] Starting {name}", flush=True)
    with log_path.open("w") as log:
      self.child = subprocess.Popen(
        command, cwd=self.work, env=self.env, stdout=log, stderr=subprocess.STDOUT
      )
      self.state["child_pid"] = self.child.pid
      self.save()
      while True:
        try:
          code = self.child.wait(timeout=15)
          break
        except subprocess.TimeoutExpired:
          with log_path.open("rb") as current:
            current.seek(max(0, log_path.stat().st_size - 65536))
            text = current.read().decode(errors="replace")
          progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
          if progress:
            stage["iteration"], stage["iterations"] = map(int, progress[-1])
          self.save()
    self.child = None
    self.state.pop("child_pid", None)
    stage.update(
      status="completed" if code == 0 else "failed", return_code=code, ended_at=now()
    )
    self.save()
    if code:
      raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
    print(f"[{now()}] Completed {name}", flush=True)

  def wait_for_gpu(self):
    """Avoid starting a stage alongside an external CUDA process."""
    uuid = subprocess.check_output(
      [
        "nvidia-smi",
        "-i",
        str(self.cfg.gpu),
        "--query-gpu=uuid",
        "--format=csv,noheader",
      ],
      text=True,
    ).strip()
    while True:
      if os.environ.get("ADAPT_G1_ALLOW_GPU_SHARING") == "1":
        self.state["gpu_sharing_allowed"] = True
        self.save()
        return
      output = subprocess.check_output(
        [
          "nvidia-smi",
          "--query-compute-apps=gpu_uuid,pid",
          "--format=csv,noheader,nounits",
        ],
        text=True,
      )
      foreign = []
      for line in output.splitlines():
        device, pid = (part.strip() for part in line.split(","))
        if device == uuid:
          try:
            if os.getpgid(int(pid)) != os.getpgrp():
              foreign.append(int(pid))
          except ProcessLookupError:
            continue
      self.state["foreign_gpu_pids"] = foreign
      self.save()
      if not foreign:
        return
      print(
        f"[{now()}] Waiting for GPU {self.cfg.gpu}; external CUDA PIDs: {foreign}",
        flush=True,
      )
      time.sleep(15)

  def train(self, name, task, experiment, iterations):
    envs = getattr(
      self.cfg, "tracker_num_envs" if name == "stage1" else "planner_num_envs", None
    )
    envs = envs or self.cfg.num_envs
    self.run(
      name,
      "mjlab.scripts.train",
      [
        task,
        "--gpu-ids",
        "[0]",
        "--env.scene.num-envs",
        envs,
        "--agent.resume",
        "False",
        "--agent.max-iterations",
        iterations,
        "--agent.save-interval",
        self.cfg.save_interval,
        "--agent.run-name",
        f"{self.root.name}_{name}",
      ],
    )
    runs = [
      path
      for path in (self.work / "logs/rsl_rl" / experiment).glob(
        f"*_{self.root.name}_{name}"
      )
      if any(path.glob("model_*.pt"))
    ]
    if len(runs) != 1:
      raise RuntimeError(f"Cannot identify unique {name} training output: {runs}")
    checkpoints = list(runs[0].glob("model_*.pt"))
    if not checkpoints:
      raise RuntimeError(f"No checkpoint saved for {name}")
    checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
    self.state["stages"][name]["checkpoint"] = str(checkpoint)
    self.save()
    return checkpoint

  def execute(self):
    if self.cfg.resume_run:
      tracker = Path(self.state["stages"]["stage1"]["checkpoint"])
    else:
      tracker = self.train(
        "stage1", "Mjlab-G1-PingPong-Stage1", "g1_stage1", self.cfg.tracker_iterations
      )
    tracker_jit = tracker.parent / "jit" / f"modeljit_{tracker.stem.split('_')[-1]}.pt"
    if not tracker_jit.is_file():
      raise FileNotFoundError(tracker_jit)
    self.env["ADAPT_G1_TRACKER"] = str(tracker_jit)
    corrected = self.root / "corrected/motions"
    if self.state["stages"].get("correction", {}).get("status") != "completed":
      self.correct(tracker_jit, corrected)
    self.env["ADAPT_G1_MOTION_DIR"] = str(corrected)
    mvae = self.root / "mvae/mvae.pt"
    if self.state["stages"].get("mvae", {}).get("status") != "completed":
      self.fit_mvae(corrected, mvae)
    self.env["ADAPT_G1_MVAE"] = str(mvae)
    self.train_planners_and_export(corrected)

  def correct(self, tracker_jit, corrected):
    self.run(
      "correction",
      "mjlab.tasks.g1_adapt.correct_motions",
      [
        "--tracker",
        tracker_jit,
        "--motion-directory",
        self.raw,
        "--output-directory",
        corrected,
        "--device",
        "cuda:0",
      ],
    )

  def fit_mvae(self, corrected, mvae):
    options = []
    profile = self.snapshot / "training_profile.json"
    if profile.is_file():
      settings = json.loads(profile.read_text()).get("mvae", {})
      allowed = {
        "rollout",
        "batch_size",
        "batches_per_epoch",
        "save_interval",
        "select_best",
        "velocity_consistency_weight",
      }
      for key, value in settings.items():
        if key not in allowed:
          raise ValueError(f"Unsupported MVAE profile option: {key}")
        options.extend(["--" + key.replace("_", "-"), value])
    self.run(
      "mvae",
      "mjlab.tasks.g1_adapt.train_mvae",
      [
        "--motion-directory",
        corrected,
        "--output",
        mvae,
        "--epochs",
        self.cfg.mvae_epochs,
        "--device",
        "cuda:0",
        *options,
      ],
    )

  def train_planners_and_export(self, corrected):
    rally = self.train(
      "rally", "Mjlab-G1-AdaPT-Rally", "g1_adapt_rally", self.cfg.rally_iterations
    )
    standing = self.train(
      "standing",
      "Mjlab-G1-AdaPT-Standing",
      "g1_adapt_standing",
      self.cfg.standing_iterations,
    )
    for name, fixed in (("evaluate", False), ("evaluate_fixed_speed", True)):
      self.run(
        name,
        "mjlab.tasks.g1_adapt.evaluate",
        [
          "--checkpoint",
          rally,
          "--episodes",
          self.cfg.evaluation_episodes,
          "--num-envs",
          "16",
          "--seed",
          "10001",
          "--fixed-speed",
          str(fixed),
          "--output",
          self.root / f"{name}.json",
          "--device",
          "cuda:0",
        ],
      )
    self.run(
      "export",
      "mjlab.tasks.g1_adapt.export",
      [
        "--rally-checkpoint",
        rally,
        "--standing-checkpoint",
        standing,
        "--ready-motion",
        sorted(corrected.glob("*.npz"))[0],
        "--output",
        self.root / "bundle",
      ],
    )
    self.state.update(
      status="completed", ended_at=now(), bundle=str(self.root / "bundle")
    )
    self.save()
    print(
      f"[{now()}] Full training pipeline completed: {self.root / 'bundle'}", flush=True
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--tracker-num-envs", type=int)
  parser.add_argument("--planner-num-envs", type=int)
  parser.add_argument("--training-profile")
  parser.add_argument("--exclusive-gpu", action="store_true")
  parser.add_argument(
    "--resume-run",
    help="Interrupted/failed run with completed Tracker and no planner checkpoint",
  )
  parser.add_argument("--tracker-iterations", type=int, default=30000)
  parser.add_argument("--mvae-epochs", type=int, default=100)
  parser.add_argument("--rally-iterations", type=int, default=30000)
  parser.add_argument("--standing-iterations", type=int, default=10000)
  parser.add_argument("--save-interval", type=int, default=1000)
  parser.add_argument("--evaluation-episodes", type=int, default=200)
  stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
  parser.add_argument(
    "--run-directory",
    default=str(REPO / "logs/g1_adapt_full" / f"{stamp}_extended27mm_gpu0"),
  )
  cfg = parser.parse_args()
  if any(
    value is not None and value <= 0
    for value in (cfg.tracker_num_envs, cfg.planner_num_envs)
  ):
    parser.error("Stage-specific environment counts must be positive")
  if any(
    value <= 0
    for value in (
      cfg.num_envs,
      cfg.tracker_iterations,
      cfg.mvae_epochs,
      cfg.rally_iterations,
      cfg.standing_iterations,
      cfg.save_interval,
      cfg.evaluation_episodes,
    )
  ):
    parser.error(
      "Training lengths, environment counts and save intervals must be positive"
    )
  lock_path = Path(f"/tmp/adapt-g1-gpu{cfg.gpu}.lock")
  inherited = os.environ.get("ADAPT_G1_GPU_LOCK_FD")
  if inherited:
    lock = os.fdopen(os.dup(int(inherited)), "a+")
    if Path(f"/proc/self/fd/{lock.fileno()}").resolve() != lock_path:
      raise ValueError("Inherited GPU lock descriptor points to the wrong lock")
  else:
    lock = lock_path.open("a+")
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      raise RuntimeError(f"GPU {cfg.gpu} is reserved by another G1 pipeline") from None
  pipeline = Pipeline(cfg)

  def terminate(_signum, _frame):
    raise KeyboardInterrupt("Training pipeline terminated")

  signal.signal(signal.SIGTERM, terminate)
  try:
    pipeline.execute()
  except BaseException as error:
    if pipeline.child is not None and pipeline.child.poll() is None:
      pipeline.child.terminate()
      pipeline.child.wait()
    pipeline.state.update(
      status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
      error=str(error),
      ended_at=now(),
    )
    pipeline.save()
    raise


if __name__ == "__main__":
  main()

"""Optional, bounded, one-way telemetry to an isolated viewer/recorder process."""

import hashlib
import json
import multiprocessing as mp
import queue
import shutil
import time
from pathlib import Path

STATE_KEYS = (
  "time",
  "wall_time",
  "state",
  "phase",
  "policy",
  "q",
  "dq",
  "root_pos",
  "root_quat",
  "root_quat_raw_mocap",
  "root_source_stamp",
  "root_age",
  "dds_state_age",
  "imu_quat",
  "target_q",
  "raw_racket_command",
  "goal_position",
  "goal_base_xy",
  "swing_active",
  "return_anchor_world",
  "return_anchor_error_m",
  "returning_to_anchor",
  "goal_velocity",
  "goal_normal",
  "time_to_strike",
  "motion_index",
  "motor_output_enabled",
)


class LiveVisualization:
  """No rendering, prediction, JSON encoding or disk IO in publish()."""

  def __init__(self, deployment, options, output, *, host, port, headless=False):
    import mujoco

    from deploy.tools.command_ghosts import command_geometry

    destination = Path(output)
    if destination.suffix == ".jsonl":
      destination = destination.parent / (destination.stem + "_diagnostics")
    self.directory = destination / "visualization"
    self.directory.mkdir(parents=True, exist_ok=False)
    model_path = self.directory / "scene.mjb"
    mujoco.mj_saveModel(deployment.model, str(model_path))
    profile = options["physics_profile_report"]
    if profile["runtime_mode"] == "real":
      shutil.copyfile(profile["source"], self.directory / "calibration.yaml")
    metadata = dict(
      version=2,
      command_ghosts=command_geometry(deployment.model),
      world_frame="hope_table",
      physics=profile,
      checkpoint=str(deployment.checkpoint),
      mujoco_version=mujoco.__version__,
      model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
      joint_names=deployment.settings["robot"]["joint_names"],
      world_translation=deployment.settings["simulation_world_translation"],
      clock="local monotonic receive time; original source stamps retained",
      prediction_semantics="auxiliary ball-only forecast, not the planner command",
    )
    (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    ctx = mp.get_context("spawn")
    self.events = ctx.Queue(maxsize=2048)
    self.stop, ready = ctx.Event(), ctx.Event()
    self.accepted = self.dropped = 0
    from deploy.tools.real_viewer import live_worker

    self.process = ctx.Process(
      target=live_worker,
      args=(str(self.directory), self.events, self.stop, ready, host, port, headless),
      name="g1-live-viewer",
      daemon=True,
    )
    self.process.start()
    deadline = time.monotonic() + 30
    while not ready.wait(0.1):
      if not self.process.is_alive() or time.monotonic() > deadline:
        self.close()
        raise RuntimeError(
          f"Viewer startup failed; see {self.directory}/worker_error.txt"
        )
    print(f"Live visualization log: {self.directory}", flush=True)

  def publish(self, kind, data, *, received=None):
    # Rows/packets contain fresh lists, never mutable robot IO arrays. Filter
    # the controller's much larger diagnostic record before enqueueing it.
    packet = (
      {k: data[k] for k in STATE_KEYS if k in data} if kind == "state" else dict(data)
    )
    try:
      self.events.put_nowait(
        dict(
          kind=kind,
          time=time.monotonic() if received is None else received,
          data=packet,
        )
      )
      self.accepted += 1
    except (queue.Full, ValueError, OSError):
      self.dropped += 1

  def close(self):
    self.stop.set()
    self.process.join(timeout=5)
    forced = self.process.is_alive()
    if forced:
      self.process.terminate()
      self.process.join(timeout=2)
    self.events.close()
    self.events.cancel_join_thread()
    (self.directory / "transport_summary.json").write_text(
      json.dumps(
        dict(
          accepted=self.accepted,
          dropped=self.dropped,
          forced_stop=forced,
          worker_exitcode=self.process.exitcode,
        ),
        indent=2,
      )
    )

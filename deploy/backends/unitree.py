"""G1 SDK2 DDS backend with read-only default and a 500 Hz command watchdog.

Uses the official unitree_sdk2py HG LowState/LowCmd API. SDK and ROS 2 are
optional dependencies, imported only for real hardware deployment.
"""

import hashlib
import json
import multiprocessing
import platform
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from deploy.backends.mujoco import MujocoBackend
from deploy.backends.runtime import SimulationIO, SimulationReset
from deploy.backends.sdk import require_sdk
from deploy.g1_pingpong.commands import CommandGuard, policy_pd_settings
from deploy.g1_pingpong.core import quaternion_multiply, rotation
from deploy.g1_pingpong.oscillation import OscillationGuard
from deploy.g1_pingpong.pd_gains import deployment_gains
from deploy.g1_pingpong.rally import RallyController
from deploy.g1_pingpong.stable import HomePosture, SmoothHandoff, StableAMP
from deploy.io.diagnostics import (
  BufferedWriter,
  CommandWriteRecorder,
  FeedbackTiming,
  LowStateRecorder,
  json_default,
)
from deploy.io.mocap import read_config
from deploy.io.transport import build_mocap

RUNTIME_REVISION = "20261016-hope-public-stop-13"


def check_motion_mode(cfg, *, allow_release=False):
  """Verify SDK low-level handover; mutation requires explicit motor startup."""
  from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
  )

  client = MotionSwitcherClient()
  client.SetTimeout(cfg.get("motion_mode_check_timeout", 3.0))
  client.Init()
  initial_mode = None
  release_attempts = 0
  while True:
    status, mode = client.CheckMode()
    if (
      status != 0 or not isinstance(mode, dict) or not isinstance(mode.get("name"), str)
    ):
      raise RuntimeError(
        f"Cannot verify robot control mode: status={status}, mode={mode}"
      )
    if initial_mode is None:
      initial_mode = dict(mode)
    if not mode["name"]:
      return {
        "status": status,
        "initial_mode": initial_mode,
        "mode": mode,
        "release_called": release_attempts > 0,
        "release_attempts": release_attempts,
      }
    if not allow_release:
      raise RuntimeError(f"Robot motion mode is not released: {mode['name']}")
    if release_attempts >= 3:
      raise RuntimeError(f"Robot motion mode did not release: {mode['name']}")
    print(f"[DDS] Releasing motion mode: {mode['name']}", flush=True)
    release_status, _ = client.ReleaseMode()
    release_attempts += 1
    if release_status != 0:
      raise RuntimeError(f"Robot motion mode release failed: status={release_status}")
    time.sleep(0.5)


# Official G1 HG SDK motor order, independent of the policy/XML order.
SDK_JOINT_NAMES = (
  [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  ]
  + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
  + [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
      "shoulder_pitch",
      "shoulder_roll",
      "shoulder_yaw",
      "elbow",
      "wrist_roll",
      "wrist_pitch",
      "wrist_yaw",
    )
  ]
)


def gravity_disagreement(imu_quat, mocap_quat, imu_to_pelvis):
  """Compare tilt in pelvis axes, allowing different IMU/mocap world headings."""
  imu_gravity = rotation(imu_to_pelvis) @ (rotation(imu_quat).T @ [0.0, 0.0, -1.0])
  mocap_gravity = rotation(mocap_quat).T @ [0.0, 0.0, -1.0]
  return float(np.degrees(np.arccos(np.clip(imu_gravity @ mocap_gravity, -1.0, 1.0))))


def root_orientation(mocap_quat, imu_quat, cfg):
  """HOPE yaw composition; position remains the calibrated mocap pelvis."""
  source = cfg.get("root_orientation_source", "mocap")
  if source == "mocap":
    return np.asarray(mocap_quat)
  if source != "imu_mocap_yaw" or imu_quat is None:
    raise ValueError("Expected mocap or imu_mocap_yaw root orientation with IMU")
  extrinsic = np.asarray(cfg["imu_to_pelvis_quaternion_wxyz"])
  imu_pelvis = quaternion_multiply(
    imu_quat, extrinsic * np.array([1.0, -1.0, -1.0, -1.0])
  )
  imu_pelvis = imu_pelvis / np.linalg.norm(imu_pelvis)

  def yaw(quat):
    r = rotation(quat)
    return np.arctan2(r[1, 0], r[0, 0])

  delta = yaw(mocap_quat) - yaw(imu_pelvis)
  return quaternion_multiply(
    np.array([np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)]), imu_pelvis
  )


class DDSBackend:
  def __init__(self, manifest, cfg, enable_motors=False):
    require_sdk()
    from unitree_sdk2py.core.channel import (
      ChannelFactoryInitialize,
      ChannelPublisher,
      ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    self.cfg, self.robot = cfg, manifest["robot"]
    self.policy_to_sdk = [SDK_JOINT_NAMES.index(n) for n in self.robot["joint_names"]]
    self.enabled = enable_motors
    self.lock, self.stop = threading.Lock(), threading.Event()
    self.latest = self.command = None
    self.diagnostic_enabled = cfg.get("record_io_diagnostics", False) or cfg.get(
      "record_command_writes", False
    )
    self.io_diagnostics = {}
    self.snapshot_diagnostics = {}
    self.pending_origin = self.command_origin = None
    self.halted = False
    self.error = None
    self.lowstate_recorder = None
    self.command_recorder = None
    self.receive_timing = FeedbackTiming()
    self.pending_context = self.command_context = {}
    self.command_sequence = self.write_sequence = 0
    self.crc = CRC()
    self.low_cmd = unitree_hg_msg_dds__LowCmd_()
    ChannelFactoryInitialize(cfg["domain_id"], cfg["network_interface"])
    if cfg.get("diagnostic_output_dir") and cfg.get("record_lowstate_hz", 0) > 0:
      self.lowstate_recorder = LowStateRecorder(
        Path(cfg["diagnostic_output_dir"]) / "lowstate",
        self.robot["joint_names"],
        cfg["record_lowstate_hz"],
      )
    if (
      enable_motors
      and cfg.get("diagnostic_output_dir")
      and cfg.get("record_command_writes", False)
    ):
      self.command_recorder = CommandWriteRecorder(
        Path(cfg["diagnostic_output_dir"]) / "command_writes",
        self.robot["joint_names"],
      )
    self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    self.subscriber.Init(self._receive, 1)
    self.publisher = None
    if enable_motors:
      self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
      self.publisher.Init()
    self.thread = threading.Thread(target=self._write_loop, daemon=True)
    self.thread.start()

  def _receive(self, msg):
    # Copy the message fields atomically; do not retain an SDK-owned object.
    callback_time = time.monotonic()
    timing = getattr(self, "receive_timing", None)
    if timing is None:
      timing = self.receive_timing = FeedbackTiming()
    receive_timing = timing.observe(callback_time, int(msg.tick))
    q = np.array([msg.motor_state[i].q for i in self.policy_to_sdk])
    dq = np.array([msg.motor_state[i].dq for i in self.policy_to_sdk])
    remote = bytes(msg.wireless_remote)
    buttons = int.from_bytes(remote[2:4], "little")
    gyro = np.array(msg.imu_state.gyroscope)
    imu_quat = np.array(msg.imu_state.quaternion)
    raw_recorder = getattr(self, "lowstate_recorder", None)
    if raw_recorder is not None and raw_recorder.due(callback_time):
      with self.lock:
        written = self.io_diagnostics.copy()
      motors = [msg.motor_state[i] for i in self.policy_to_sdk]
      raw_recorder.write(
        (
          callback_time,
          int(msg.tick),
          int(msg.mode_machine),
          buttons,
          q,
          dq,
          [float(m.tau_est) for m in motors],
          gyro,
          imu_quat,
          [int(m.motorstate) for m in motors],
          [float(m.vol) for m in motors],
          [list(m.temperature) for m in motors],
          written.get("last_written_origin", float("nan")),
          int(written.get("writer_halted", False)),
          receive_timing["feedback_sequence"],
          written.get("write_sequence", 0),
          written.get("last_written_command_id", 0),
          written.get("write_started_time", float("nan")),
          written.get("write_completed_time", float("nan")),
        )
      )
    with self.lock:
      if not all(np.isfinite(value).all() for value in (q, dq, gyro, imu_quat)):
        self.error = "G1 lowstate contains non-finite feedback"
        self.halted = True
        return
      if imu_quat.shape != (4,) or np.linalg.norm(imu_quat) < 1e-8:
        self.error, self.halted = "Invalid pelvis IMU quaternion", True
        return
      self.latest = (
        q,
        dq,
        gyro,
        buttons,
        int(msg.mode_machine),
        imu_quat / np.linalg.norm(imu_quat),
        callback_time,
      )
      if self.diagnostic_enabled:
        self.io_diagnostics.update(
          **receive_timing,
          state_source_tick=int(msg.tick),
          state_callback_time=self.latest[-1],
          tau_est=[float(msg.motor_state[i].tau_est) for i in self.policy_to_sdk],
        )
      if self.enabled and buttons & ((1 << 3) | (1 << 9) | (1 << 10)):
        self.halted = True

  def feedback(self):
    with self.lock:
      if self.error is not None:
        if (
          not self.cfg.get("latch_damping_on_fault", False)
          or self.latest is None
          or self.error.startswith("DDS writer failed")
        ):
          raise RuntimeError(self.error)
        self.halted = True
        self.io_diagnostics["halt_reason"] = self.error
      if self.latest is None:
        return None
      result = self.latest
      if time.monotonic() - result[-1] > self.cfg["state_timeout"]:
        self.halted = True
        if not self.cfg.get("latch_damping_on_fault", False):
          raise TimeoutError("G1 lowstate expired")
        self.io_diagnostics["halt_reason"] = "G1 lowstate expired"
      if self.enabled and result[4] != self.cfg["expected_mode_machine"]:
        self.halted = True
        raise ValueError("G1 mode_machine differs from configured waist mode")
      if self.diagnostic_enabled:
        self.snapshot_diagnostics = self.io_diagnostics.copy()
      return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in result)

  def note_command_origin(self, generated):
    """Associate a queued command with its original policy-process timestamp."""
    with self.lock:
      self.pending_origin = generated

  def note_command_context(self, context):
    with self.lock:
      self.pending_context = dict(context)

  def diagnostics(self):
    with self.lock:
      result = (self.snapshot_diagnostics or self.io_diagnostics).copy()
    raw_recorder = getattr(self, "lowstate_recorder", None)
    if raw_recorder is not None:
      result["lowstate_recording"] = raw_recorder.diagnostics()
    command_recorder = getattr(self, "command_recorder", None)
    if command_recorder is not None:
      result["command_write_recording"] = command_recorder.diagnostics()
    return result

  def send(self, target, kp=None, kd=None):
    kp = np.asarray(self.robot["kp"] if kp is None else kp)
    kd = np.asarray(self.robot["kd"] if kd is None else kd)
    if any(
      np.asarray(a).shape != (29,) or not np.isfinite(a).all() for a in (target, kp, kd)
    ):
      raise ValueError("Invalid DDS command")
    if np.any(kp < 0) or np.any(kd < 0):
      raise ValueError("Negative motor gains")
    with self.lock:
      if self.halted:
        raise RuntimeError("DDS backend is halted")
      self.command = (target.copy(), kp.copy(), kd.copy(), time.monotonic())
      self.command_origin = (
        self.pending_origin if self.pending_origin is not None else self.command[-1]
      )
      self.pending_origin = None
      self.command_sequence = getattr(self, "command_sequence", 0) + 1
      self.command_context = dict(getattr(self, "pending_context", {}))
      self.command_context.setdefault("command_id", self.command_sequence)
      self.pending_context = {}
      self.command_context["accepted_time"] = self.command[-1]
      self.last_submit_diagnostics = dict(
        command_id=self.command_context["command_id"],
        command_generated_time=self.command_origin,
        enqueue_completed_time=self.command[-1],
        command_queue_put_succeeded=True,
      )
      if self.diagnostic_enabled:
        self.io_diagnostics.update(
          command_origin_time=self.command_origin,
          command_accepted_time=self.command[-1],
          command_queue_age=self.command[-1] - self.command_origin,
        )

    return self.command_origin

  def halt(self):
    with self.lock:
      self.halted = True

  def _write_loop(self):
    deadline = time.monotonic()
    try:
      while not self.stop.is_set():
        with self.lock:
          feedback, command = self.latest, self.command
          command_origin = self.command_origin
          context = dict(getattr(self, "command_context", {}))
          writer_tick = self.io_diagnostics.get("state_source_tick", -1)
          now = time.monotonic()
          if command is not None and now - command[-1] > self.cfg["command_timeout"]:
            self.halted = True
            self.io_diagnostics["halt_reason"] = "Motor command expired"
          if feedback is not None and now - feedback[-1] > self.cfg["state_timeout"]:
            self.halted = True
            self.io_diagnostics["halt_reason"] = "G1 lowstate expired"
          halted = self.halted
        if self.enabled and feedback is not None and command is not None:
          prepare_started = time.monotonic()
          q, _, _, _, mode = feedback[:5]
          if mode != self.cfg["expected_mode_machine"]:
            raise ValueError("Unexpected robot mode")
          target, kp, kd, _ = command
          if halted:
            damping = (
              np.asarray(self.robot["kd"])
              if self.cfg.get("halt_with_policy_kd", False)
              else np.full(29, self.cfg["halt_kd"])
            )
            target, kp, kd = q, np.zeros(29), damping
          self.low_cmd.mode_pr = 0
          self.low_cmd.mode_machine = mode
          for policy_index, sdk_index in enumerate(self.policy_to_sdk):
            motor = self.low_cmd.motor_cmd[sdk_index]
            motor.mode, motor.q, motor.dq, motor.tau = (
              1,
              float(target[policy_index]),
              0.0,
              0.0,
            )
            motor.kp, motor.kd = float(kp[policy_index]), float(kd[policy_index])
          self.low_cmd.crc = self.crc.Crc(self.low_cmd)
          assert self.publisher is not None
          write_started = time.monotonic() if self.diagnostic_enabled else None
          write_result = self.publisher.Write(self.low_cmd)
          write_completed = time.monotonic()
          if self.diagnostic_enabled:
            with self.lock:
              previous_write = self.io_diagnostics.get("write_started_time")
              first_write = (
                self.io_diagnostics.get("last_written_origin") != command_origin
              )
              self.write_sequence = getattr(self, "write_sequence", 0) + 1
              if first_write:
                self.io_diagnostics["command_first_write_time"] = write_started
                self.io_diagnostics["command_to_first_write_age"] = (
                  write_started - command_origin
                )
              self.io_diagnostics.update(
                write_started_time=write_started,
                write_completed_time=write_completed,
                write_sequence=self.write_sequence,
                last_written_command_id=context.get("command_id", 0),
                last_written_origin=command_origin,
                writer_halted=halted,
                write_result=write_result
                if isinstance(write_result, (bool, int, float, type(None)))
                else type(write_result).__name__,
                write_interval_max=max(
                  self.io_diagnostics.get("write_interval_max", 0.0),
                  write_started - previous_write if previous_write is not None else 0.0,
                ),
              )
            journal = getattr(self, "command_recorder", None)
            if journal is not None:
              journal.write(
                (
                  command_origin,
                  context.get("accepted_time", command[-1]),
                  context.get("feedback_callback_time", float("nan")),
                  context.get("feedback_consumed_time", float("nan")),
                  context.get("feedback_packet_enqueued_time", float("nan")),
                  context.get("feedback_packet_dequeued_time", float("nan")),
                  prepare_started,
                  deadline,
                  write_started,
                  write_completed,
                  feedback[-1],
                  write_started - previous_write
                  if previous_write is not None
                  else float("nan"),
                  self.write_sequence,
                  context.get("command_id", 0),
                  context.get("control_cycle", 0),
                  context.get("feedback_sequence", 0),
                  context.get("source_tick", -1),
                  writer_tick,
                  int(first_write),
                  int(halted),
                  int(write_result) if isinstance(write_result, (bool, int)) else -1,
                  target.copy(),
                  kp.copy(),
                  kd.copy(),
                  context.get("command_dequeued_time", float("nan")),
                )
              )
        deadline += 0.002
        if deadline < time.monotonic() - 0.002:
          deadline = time.monotonic()
        self.stop.wait(max(deadline - time.monotonic(), 0))
    except Exception as exc:
      self.error = f"DDS writer failed: {exc}"
      self.halted = True

  def close(self):
    with self.lock:
      self.halted = True
    if self.enabled:
      time.sleep(0.1)
    self.stop.set()
    self.thread.join(timeout=2)
    with self.lock:
      self.snapshot_diagnostics = {}
    # SDK channel Close exists on current SDK2; tolerate older releases.
    for channel in (self.subscriber, self.publisher):
      if channel is not None and hasattr(channel, "Close"):
        channel.Close()
    raw_recorder = getattr(self, "lowstate_recorder", None)
    if raw_recorder is not None:
      raw_recorder.close()
    journal = getattr(self, "command_recorder", None)
    if journal is not None:
      journal.close()


def _replace_latest(channel, item):
  """Bounded queue; never block robot communication on a slow policy reader."""
  try:
    channel.put_nowait(item)
    return True
  except queue.Full:
    try:
      channel.get_nowait()
    except queue.Empty:
      pass
    try:
      channel.put_nowait(item)
      return True
    except queue.Full:
      return False


def _dds_process(manifest, cfg, enabled, commands, feedback, stop, halt, factory):
  backend = None
  last_command = None
  packet_sequence = 0
  try:
    backend = factory(manifest, cfg, enabled)
    while not stop.is_set():
      if halt.is_set():
        backend.halt()
      try:
        target, kp, kd, generated, *metadata = commands.get(
          timeout=cfg.get("io_poll_seconds", 0.001)
        )
        dequeued_time = time.monotonic()
        # A queued command must not gain a new lease just because the IO
        # process finally received it after a long scheduling delay.
        if time.monotonic() - generated > cfg["command_timeout"]:
          backend.halt()
        elif not backend.halted:
          if hasattr(backend, "note_command_origin"):
            backend.note_command_origin(generated)
          if metadata and hasattr(backend, "note_command_context"):
            backend.note_command_context(
              {
                **metadata[0],
                "command_dequeued_time": dequeued_time,
              }
            )
          backend.send(target, kp, kd)
          last_command = generated
      except queue.Empty:
        pass
      if (
        last_command is not None
        and time.monotonic() - last_command > cfg["command_timeout"]
      ):
        backend.halt()
      packet = {"state": backend.feedback(), "halted": backend.halted}
      if (
        cfg.get("record_io_diagnostics", False)
        or cfg.get("record_command_writes", False)
      ) and hasattr(backend, "diagnostics"):
        packet["diagnostics"] = backend.diagnostics()
      packet_sequence += 1
      packet["packet_sequence"] = packet_sequence
      packet["enqueued_time"] = time.monotonic()
      _replace_latest(feedback, packet)
  except Exception as exc:
    _replace_latest(feedback, {"error": f"DDS process failed: {exc}", "halted": True})
  finally:
    if backend is not None:
      backend.close()
      if hasattr(backend, "diagnostics"):
        _replace_latest(
          feedback,
          {"state": None, "halted": True, "diagnostics": backend.diagnostics()},
        )


class DDSProcessBackend:
  """Run the SDK and its writer/watchdog outside the inference process."""

  def __init__(self, manifest, cfg, enable_motors=False, *, factory=DDSBackend):
    self.cfg, self.robot = cfg, manifest["robot"]
    poll = cfg.get("io_poll_seconds", 0.001)
    if not np.isfinite(poll) or not 0 < poll <= min(
      cfg["state_timeout"], cfg["command_timeout"]
    ):
      raise ValueError("DDS polling period must fit within watchdog leases")
    context = multiprocessing.get_context("spawn")
    self.commands = context.Queue(maxsize=1)
    self.states = context.Queue(maxsize=2)
    self.stop, self.halt_event = context.Event(), context.Event()
    self.latest, self.halted = None, False
    self.io_diagnostics = {}
    self.pending_context = {}
    self.command_sequence = 0
    self.last_submit_diagnostics = {}
    self.process = context.Process(
      target=_dds_process,
      args=(
        manifest,
        cfg,
        enable_motors,
        self.commands,
        self.states,
        self.stop,
        self.halt_event,
        factory,
      ),
      name="g1-dds-bridge",
    )
    self.process.start()

  def feedback(self):
    for index in range(32):
      try:
        # multiprocessing.Queue publishes through a feeder thread. The IO
        # producer also removes an old packet before replacing it. An empty
        # nonblocking read can therefore be transient even with fresh SDK
        # callbacks. Give the first packet one IO poll to arrive rather than
        # silently reusing a full policy cycle (20ms) old snapshot. Subsequent
        # reads only drain what is already available; the wait is bounded once.
        packet = (
          self.states.get(timeout=self.cfg.get("io_poll_seconds", 0.001))
          if index == 0
          else self.states.get_nowait()
        )
      except queue.Empty:
        break
      self.halted |= packet["halted"]
      if "error" in packet:
        self.halt()
        raise RuntimeError(packet["error"])
      self.io_diagnostics = packet.get("diagnostics", self.io_diagnostics)
      if packet["state"] is not None:
        self.latest = packet["state"]
        self.io_diagnostics.update(
          feedback_packet_sequence=packet.get("packet_sequence"),
          feedback_packet_enqueued_time=packet.get("enqueued_time"),
          feedback_packet_dequeued_time=time.monotonic(),
        )
    if not self.process.is_alive():
      self.halt()
      raise RuntimeError(f"DDS process exited (code {self.process.exitcode})")
    if (
      self.latest is not None
      and time.monotonic() - self.latest[-1] > self.cfg["state_timeout"]
    ):
      self.halt()
      if not self.cfg.get("latch_damping_on_fault", False):
        raise TimeoutError("DDS process feedback expired")
      self.io_diagnostics["halt_reason"] = "DDS process feedback expired"
    self.io_diagnostics["feedback_consumed_time"] = time.monotonic()
    return self.latest

  def diagnostics(self):
    return self.io_diagnostics.copy()

  def note_command_context(self, context):
    self.pending_context = dict(context)

  def send(self, target, kp=None, kd=None):
    kp = np.asarray(self.robot["kp"] if kp is None else kp)
    kd = np.asarray(self.robot["kd"] if kd is None else kd)
    values = [np.asarray(a, dtype=float) for a in (target, kp, kd)]
    if any(a.shape != (29,) or not np.isfinite(a).all() for a in values):
      raise ValueError("Invalid process motor command")
    if np.any(kp < 0) or np.any(kd < 0):
      raise ValueError("Negative motor gains")
    if self.halted or not self.process.is_alive():
      raise RuntimeError("DDS process is halted or unavailable")
    generated = time.monotonic()
    self.command_sequence += 1
    context = self.pending_context
    self.pending_context = {}
    context["command_id"] = self.command_sequence
    queued = _replace_latest(self.commands, (*values, generated, context))
    self.last_submit_diagnostics = dict(
      command_id=self.command_sequence,
      command_generated_time=generated,
      enqueue_completed_time=time.monotonic(),
      command_queue_put_succeeded=queued,
    )
    return generated

  def halt(self):
    self.halted = True
    self.halt_event.set()

  def close(self):
    self.halt()
    self.stop.set()
    self.process.join(timeout=8)
    if self.process.is_alive():
      raise RuntimeError("DDS process failed to stop; halt remains requested")
    for _ in range(32):
      try:
        packet = self.states.get_nowait()
      except queue.Empty:
        break
      self.io_diagnostics = packet.get("diagnostics", self.io_diagnostics)
    for channel in (self.commands, self.states):
      channel.close()
      channel.cancel_join_thread()


def run_real(deployment, real_cfg, mocap_config, **kwargs):
  """Legacy mjlab receiving task; keep its Stage 2 contract explicit."""
  if deployment.settings["stage"] != 2:
    raise ValueError("Receiving rally requires a Stage 2 checkpoint")
  return run_control_loop(deployment, real_cfg, mocap_config, **kwargs)


def run_control_loop(
  deployment,
  real_cfg,
  mocap_config,
  *,
  enable_motors=False,
  motion_index=0,
  duration=60.0,
  output=None,
  simulation_config="deploy/configs/mujoco.yaml",
  motion_select="auto",
  motion_allow=None,
  simulation_io=None,
  controller_factory=RallyController,
):
  """Run a task controller through shared operator, IO and safety machinery.

  Task entry points validate their own policy contracts. The controller owns
  observations, goals, action history, selection and recovery; this loop owns
  feedback, AMP handoff, guarded command submission and damping on faults.
  """
  if simulation_io is not None and not isinstance(simulation_io, SimulationIO):
    raise TypeError("simulation_io must be an explicit SimulationIO adapter")
  clock = simulation_io.clock if simulation_io is not None else time
  if simulation_io is None:
    from deploy.g1_pingpong.hope_safety import real_stop_settings

    real_cfg = real_stop_settings(real_cfg)
  cfg = deployment.settings
  if enable_motors and cfg.get("synthetic", False) and simulation_io is None:
    raise ValueError("Synthetic selfcheck policies cannot drive real motors")
  trained_robot_contract = cfg["robot"]
  cfg = policy_pd_settings(cfg, real_cfg)
  mocap_cfg, _ = read_config(mocap_config)
  root_grace = None
  if real_cfg.get("latch_damping_on_fault", False):
    from deploy.g1_pingpong.hope_safety import RootPoseGrace

    mocap_cfg["root_stale_timeout"] = real_cfg["root_fresh_seconds"]
    root_grace = RootPoseGrace(
      real_cfg["root_fresh_seconds"], real_cfg["root_grace_seconds"]
    )
  options = yaml.safe_load(Path(simulation_config).read_text())
  engine = controller_factory(
    deployment,
    real_cfg,
    options,
    home_index=motion_index,
    allow=motion_allow,
    selection=motion_select,
    auto_start=False,
    auto_arm=False,
  )
  if simulation_io is not None:
    simulation_io.on_controller(engine)
  print(engine.describe(), flush=True)
  lifecycle = engine.lifecycle
  kinematics = MujocoBackend(deployment.model, cfg)
  guard = CommandGuard(
    cfg["robot"]["joint_limits"], real_cfg["guard"], cfg["robot"]["joint_names"]
  )
  oscillation = OscillationGuard(
    cfg["robot"]["joint_names"],
    real_cfg["guard"].get("oscillation", {}),
    cfg["control_dt"],
  )
  stable_cfg = real_cfg.get("stable_policy", {})
  stable = None
  if enable_motors and stable_cfg.get("enabled", False):
    stable = StableAMP(
      cfg["robot"]["joint_names"], SDK_JOINT_NAMES, stable_cfg["asset_dir"]
    )
    stable.kp, stable.kd = deployment_gains(
      cfg["robot"]["joint_names"], profile="stable", scales=stable_cfg.get("pd", {})
    )
    lifecycle.initial_q = stable.default_q.copy()
  last_command, handoff = None, None
  policy_stable_since = None
  home_posture = None
  entry_ready = False
  real_cfg = real_cfg.copy()
  if output and (
    real_cfg.get("record_lowstate_hz", 0) > 0
    or real_cfg.get("record_command_writes", False)
  ):
    destination = Path(output)
    diagnostic_dir = (
      destination.parent / (destination.stem + "_diagnostics")
      if destination.suffix == ".jsonl"
      else destination
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    real_cfg["diagnostic_output_dir"] = str(diagnostic_dir.resolve())
    mocap_cfg = {**mocap_cfg, "diagnostic_output_dir": str(diagnostic_dir.resolve())}

  def begin_handoff(now, *, to_stable=False):
    return SmoothHandoff(
      last_command,
      now,
      stable_cfg.get("stable_transition_seconds", stable_cfg["transition_seconds"])
      if to_stable
      else stable_cfg["transition_seconds"],
      cfg["control_dt"],
      stable_cfg["max_target_velocity"] if to_stable else float("inf"),
      timeout=stable_cfg.get("handoff_timeout", 5.0),
    )

  mocap, robot, recorder = None, None, None
  summary_path = None
  started = clock.monotonic()
  feedback = None
  pending_target = None
  last_row = None
  cycle_index = 0
  previous_cycle = None
  previous_record_seconds = 0.0
  loop_period = None
  loop_lateness = 0.0
  fault_snapshot = None
  feedback_context = {}
  previous_feedback_callback = None
  limit_observations = []
  limit_counts, limit_printed = {}, {}

  def physical_limit(kind, error):
    if simulation_io is None or not simulation_io.observe_physical_limits:
      raise error
    limit_observations.append({"kind": kind, "error": str(error)})
    limit_counts[kind] = limit_counts.get(kind, 0) + 1
    now = clock.monotonic()
    if now - limit_printed.get(kind, float("-inf")) >= 1:
      print(f"[MuJoCo observe {now:.2f}s] {error}; policy continues", flush=True)
      limit_printed[kind] = now

  def check_command(target, q, dq):
    try:
      guard.check(target, q, dq, cfg["control_dt"])
    except ValueError as exc:
      # Nonfinite/malformed signals cannot be passed to a policy or actuator.
      if str(exc) == "Invalid motor feedback/command":
        raise
      physical_limit("command_guard", exc)
      guard.previous, guard.previous_q = target.copy(), q.copy()

  if simulation_io is not None and simulation_io.observe_physical_limits:
    engine.physical_limit_observer = physical_limit

  def send_command(target, kp, kd):
    context = {**feedback_context, "control_cycle": cycle_index}
    if hasattr(robot, "note_command_context"):
      robot.note_command_context(context)
    return robot.send(target, kp, kd)

  def file_identity(path):
    if path is None or not Path(path).is_file():
      return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
      for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}

  def record_row(row):
    nonlocal last_row, previous_record_seconds
    row.update(
      cycle_index=cycle_index,
      wall_time=clock.time(),
      loop_period_seconds=loop_period,
      loop_start_lateness_seconds=loop_lateness,
      previous_record_enqueue_seconds=previous_record_seconds,
    )
    latency = dict(feedback_context)
    if row.get("command_generated_time") is not None:
      submit = getattr(robot, "last_submit_diagnostics", {})
      latency.update(submit)
      latency["callback_to_generation_seconds"] = (
        row["command_generated_time"] - feedback_context["feedback_callback_time"]
      )
    row["latency"] = latency
    if simulation_io is not None:
      row["robot_command_scope"] = "simulation"
      row["physical_limit_observations"] = list(limit_observations)
    if recorder:
      before = clock.monotonic()
      payload = json.dumps(row, default=json_default)
      last_row = payload
      recorder.write(payload + "\n")
      previous_record_seconds = clock.monotonic() - before
    else:
      last_row = row
    if simulation_io is not None:
      simulation_io.on_record(row)

  result = {
    "mode": "mujoco" if simulation_io is not None else "real",
    "actual_robot_commands_sent": None if simulation_io is None else False,
    "simulation": simulation_io.metadata if simulation_io is not None else None,
    "physical_limit_observation_counts": limit_counts,
    "checkpoint": str(deployment.checkpoint),
    "task": cfg["task"],
    "controller": type(engine).__module__ + "." + type(engine).__name__,
    "home_motion": motion_index,
    "motion_select": motion_select,
    "motion_allow": engine.candidate_indices,
    "events": engine.events,
    "phases": [],
    "failure": None,
    "termination": "duration",
    "motor_output_enabled": enable_motors,
    "commands_sent": 0,
    "robot_command_contract": cfg["robot"],
    "trained_robot_command_contract": trained_robot_contract,
    "real_runtime_config": real_cfg,
    "rally_runtime_config": options,
    "mocap_runtime_config": mocap_cfg,
    "control_world_frame": cfg.get("world_frame", mocap_cfg["world_frame"]),
    "simulation_world_translation": cfg.get("simulation_world_translation"),
    "control_dt": cfg["control_dt"],
    "physics_dt": cfg["physics_dt"],
    "diagnostic_schema_version": 5,
    "latency_clock_contract": {
      "host_clock": (
        "time.monotonic, shared across local processes"
        if simulation_io is None
        else "virtual adapter clock; not measured host latency"
      ),
      "device_tick": (
        "raw uint32 device counter; not host-clock synchronized"
        if simulation_io is None
        else "synthetic counter; not a robot measurement"
      ),
      "policy_period_seconds": cfg["control_dt"],
      "dds_writer_period_seconds": 0.002,
      "actuation_acknowledgement": False,
      "command_write_journal": real_cfg.get("record_command_writes", False),
    },
    "policy_files": {
      "execution_pd": file_identity(
        Path(__file__).resolve().parent.parent / "g1_pingpong/pd_gains.py"
      ),
      "checkpoint": file_identity(deployment.checkpoint),
      "tracker": file_identity(deployment.tracker_path),
      "motions": [
        file_identity(deployment.motions[i])
        for i in sorted(set([motion_index, *engine.candidate_indices]))
      ],
    },
    "stable_policy_metadata": getattr(stable, "metadata", None),
    "stable_execution_pd": {
      "source": "deploy.g1_pingpong.pd_gains.JOINT_PD:stable",
      "kp": stable.kp.tolist(),
      "kd": stable.kd.tolist(),
      "scales": stable_cfg.get("pd", {}),
    }
    if stable is not None
    else None,
    "software": {
      "runtime_revision": RUNTIME_REVISION,
      "python": sys.version,
      "platform": platform.platform(),
      "numpy": np.__version__,
      "mujoco": mujoco.__version__,
    },
  }
  try:
    if simulation_io is None:
      mocap = build_mocap(mocap_cfg)
      backend_type = (
        DDSProcessBackend if real_cfg.get("io_process", True) else DDSBackend
      )
      robot = backend_type(cfg, real_cfg, enable_motors)
    else:
      mocap = simulation_io.mocap_factory(mocap_cfg)
      robot = simulation_io.robot_factory(cfg, real_cfg, enable_motors)
    if output:
      output = Path(output)
      if output.suffix == ".jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path = output.with_suffix(".summary.json")
        recorder = BufferedWriter(output)
      else:
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary.json"
        recorder = BufferedWriter(output / "trace.jsonl")
    started = deadline = clock.monotonic()
    next_diagnostic = started
    controls = (
      "START=default PD A=stable UP=pingpong DOWN=stable SELECT/B/X=halt"
      if stable is not None
      else "START=PD home pose A=policy/receive SELECT/B/X=halt"
    )
    print(
      ("MuJoCo command output " if simulation_io is not None else "DDS command output ")
      + ("ENABLED; " + controls if enable_motors else "DISABLED (read-only)")
    )
    print(
      f"Runtime revision: {RUNTIME_REVISION}; policy PD: {real_cfg.get('policy_pd', {})}",
      flush=True,
    )
    if real_cfg.get("latch_damping_on_fault", False):
      print(
        f"Stop policy: {real_cfg.get('stop_policy')}; "
        f"state/command lease={real_cfg['state_timeout']}/{real_cfg['command_timeout']}s; "
        f"tilt={real_cfg['max_tilt_degrees']}deg; faults latch policy-Kd damping",
        flush=True,
      )
    while clock.monotonic() - started < duration:
      limit_observations.clear()
      now = clock.monotonic()
      cycle_index += 1
      loop_period = now - previous_cycle if previous_cycle is not None else None
      previous_cycle = now
      loop_lateness = max(now - deadline, 0.0)
      pending_target = None
      stable_step_debug = None
      stable_compute_seconds = None
      control_work_started = now
      feedback_read_started = clock.monotonic()
      feedback = robot.feedback()
      feedback_read_completed = clock.monotonic()
      root = mocap.root.get(now)
      if root_grace is not None:
        root = root_grace.resolve(root, now)
      if robot.halted:
        if real_cfg.get("latch_damping_on_fault", False):
          reason = robot.diagnostics().get(
            "halt_reason", "IO watchdog or operator stop"
          )
          latest = getattr(robot, "latest", feedback)
          if latest is None or not latest[3] & ((1 << 3) | (1 << 9) | (1 << 10)):
            raise RuntimeError(f"Robot IO safety halt: {reason}")
        lifecycle.halt()
        result["termination"] = "halt"
        break
      if feedback is None:
        if now - started > real_cfg["connection_timeout"]:
          raise TimeoutError("No G1 lowstate received")
        clock.sleep(0.02)
        deadline = clock.monotonic()
        continue
      q, dq, gyro, buttons = feedback[:4]
      imu_quat = feedback[5] if len(feedback) == 7 else None
      io_diagnostics = robot.diagnostics() if hasattr(robot, "diagnostics") else {}
      feedback_context = dict(
        feedback_callback_time=feedback[-1],
        feedback_consumed_time=feedback_read_completed,
        feedback_read_started_time=feedback_read_started,
        feedback_read_completed_time=feedback_read_completed,
        feedback_read_seconds=feedback_read_completed - feedback_read_started,
        callback_to_consumption_seconds=feedback_read_completed - feedback[-1],
        feedback_reused=feedback[-1] == previous_feedback_callback,
        feedback_sequence=io_diagnostics.get("feedback_sequence", 0),
        source_tick=io_diagnostics.get("state_source_tick", -1),
      )
      for key in (
        "feedback_packet_sequence",
        "feedback_packet_enqueued_time",
        "feedback_packet_dequeued_time",
      ):
        if io_diagnostics.get(key) is not None:
          feedback_context[key] = io_diagnostics[key]
      previous_feedback_callback = feedback[-1]
      gravity_error = (
        gravity_disagreement(
          imu_quat, root[1], real_cfg["imu_to_pelvis_quaternion_wxyz"]
        )
        if imu_quat is not None and root is not None
        else None
      )
      if not enable_motors:
        estimate = mocap.ball.get(now)
        row = {
          "time": now,
          "state": "READ_ONLY",
          "io": io_diagnostics,
          "mocap_imu_gravity_error_degrees": gravity_error,
          "imu_quat": imu_quat.tolist() if imu_quat is not None else None,
          "root_age": now - root[3] if root else None,
          "motor_output_enabled": False,
          "q": q.tolist(),
          "dq": dq.tolist(),
          "gyro": gyro.tolist(),
          "buttons": buttons,
          "mode_machine": feedback[4],
          "mode_matches_config": feedback[4] == real_cfg["expected_mode_machine"],
          "dds_state_age": now - feedback[-1],
          "root_valid": root is not None,
          "ball_valid": estimate is not None,
          "root_pos": root[0].tolist() if root else None,
          "root_quat": root[1].tolist() if root else None,
          "ball_position": estimate.position.tolist() if estimate else None,
          "ball_velocity": estimate.velocity.tolist() if estimate else None,
          "target_q": None,
        }
        if hasattr(mocap, "diagnostics"):
          row["mocap"] = mocap.diagnostics()
        record_row(row)
        if now >= next_diagnostic:
          print(json.dumps(row))
          next_diagnostic = now + 1.0
        deadline += cfg["control_dt"]
        clock.sleep(max(deadline - clock.monotonic(), 0))
        continue
      pressed = lifecycle.edges(buttons)
      if pressed & ((1 << 3) | (1 << 9) | (1 << 10)):
        lifecycle.halt()
        result["termination"] = "halt"
        break
      if lifecycle.state == "WAIT" and pressed & (1 << 2):
        limits = np.asarray(cfg["robot"]["joint_limits"])
        if np.any(lifecycle.initial_q < limits[:, 0]) or np.any(
          lifecycle.initial_q > limits[:, 1]
        ):
          raise ValueError("Home first-frame posture outside joint limits")
        lifecycle.position(q, now)
      # No mocap or learned policy is needed while suspended and positioning.
      # An early/held A is ignored: require a fresh press after POSE_HOLD.
      if stable is not None:
        if imu_quat is None:
          raise ValueError("Stable policy requires pelvis IMU quaternion from DDS")
        # DOWN may interrupt any ping-pong phase; transition starts from the
        # last command actually sent, including gains. It does not wait for mocap.
        back_to_stable = (
          lifecycle.state
          in (
            "ARMED",
            "EXECUTE",
            "RECOVER",
            "READY",
            "PREPARE",
            "POLICY_POSTURE",
          )
          or (lifecycle.state == "STABLE" and entry_ready)
        ) and pressed & (1 << 14)
        if (lifecycle.state == "POSE_HOLD" and pressed & (1 << 8)) or back_to_stable:
          if back_to_stable:
            engine.reset_manual_policy(now)
            oscillation.reset()
            home_posture = None
            entry_ready = False
          stable.reset()
          handoff = begin_handoff(now, to_stable=True)
          lifecycle.state = "TO_STABLE"
        if lifecycle.state == "STABLE" and not entry_ready and pressed & (1 << 12):
          goal = engine.library.motions[engine.home_index].q[0]
          limits = np.asarray(cfg["robot"]["joint_limits"])
          if np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
            raise ValueError("Home preparation pose outside joint limits")
          capture_anchor = getattr(engine, "capture_return_anchor", None)
          if capture_anchor is not None:
            # Capture the operator's actual UP position, before AMP preposition
            # can move the base. The task retains the first anchor on re-entry.
            capture_anchor(
              None if root is None or (root_grace is not None and root_grace.frozen)
              else root[0],
              now, source="up_button",
            )
          home_posture = HomePosture(
            cfg["robot"]["joint_names"],
            last_command[0],
            goal,
            now,
            stable_cfg.get("preposition_seconds", 1.5),
            stable_cfg.get("preposition_hold_seconds", 0.5),
            stable_cfg.get("preposition_max_pose_error", 0.15),
          )
          lifecycle.state = "POLICY_POSTURE"
        enter_policy = entry_ready
      else:
        enter_policy = lifecycle.state == "POSE_HOLD" and bool(pressed & (1 << 8))
      if (
        lifecycle.state
        in ("WAIT", "POSITIONING", "POSE_HOLD", "TO_STABLE", "STABLE", "POLICY_POSTURE")
        and not enter_policy
      ):
        kp, kd = (
          (stable.kp, stable.kd)
          if stable is not None
          else (np.asarray(cfg["robot"]["kp"]), np.asarray(cfg["robot"]["kd"]))
        )
        target = None
        raw_target = None
        policy_seconds = None
        command_generated_time = None
        if lifecycle.state in ("TO_STABLE", "STABLE", "POLICY_POSTURE"):
          tilt = np.degrees(np.arccos(np.clip(rotation(imu_quat)[2, 2], -1, 1)))
          if tilt > real_cfg["max_tilt_degrees"]:
            physical_limit(
              "imu_tilt", ValueError("Robot IMU tilt exceeds configured limit")
            )
          policy_started = clock.monotonic()
          target = stable.step(q, dq, gyro, imu_quat)
          policy_seconds = clock.monotonic() - policy_started
          stable_step_debug = getattr(stable, "debug", {})
          raw_target = target.copy()
          if lifecycle.state == "POLICY_POSTURE":
            target = home_posture.command(target, now)
            ready = home_posture.ready(
              q,
              dq,
              gyro,
              now,
              real_cfg["standing"]["max_joint_velocity"],
              real_cfg["standing"]["max_gyro"],
              measured=engine.standing.readiness_mode == "measured",
            )
            if ready:
              entry_ready = True
            elif (
              stable_cfg.get("preposition_timeout", 10.0) is not None
              and now - home_posture.started
              > home_posture.seconds + stable_cfg.get("preposition_timeout", 10.0)
            ):
              raise TimeoutError("Home preparation did not stabilize")
          if handoff is not None:
            (target, kp, kd), done = handoff.apply((target, kp, kd), now)
            if done:
              handoff = None
              lifecycle.state = "STABLE"
        elif lifecycle.state != "WAIT":
          target = lifecycle.posture_target(now)
        if target is not None:
          pending_target = target.copy()
          check_command(target, q, dq)
          command_generated_time = send_command(target, kp, kd)
          last_command = (target.copy(), kp.copy(), kd.copy())
          result["commands_sent"] += 1
        row = {
          "time": now,
          "state": lifecycle.state,
          "phase": engine.phase,
          "q": q.tolist(),
          "dq": dq.tolist(),
          "buttons": buttons,
          "motor_output_enabled": True,
          "io": io_diagnostics,
          "command_generated_time": command_generated_time,
          "mocap_imu_gravity_error_degrees": gravity_error,
          "control_work_seconds": clock.monotonic() - control_work_started,
          "policy_enabled": lifecycle.state
          in ("TO_STABLE", "STABLE", "POLICY_POSTURE"),
          "policy": "stable_amp"
          if lifecycle.state in ("TO_STABLE", "STABLE", "POLICY_POSTURE")
          else "joint_pd",
          "stable_policy": stable_step_debug,
          "home_posture": home_posture.debug
          if lifecycle.state == "POLICY_POSTURE"
          else None,
          "policy_compute_seconds": policy_seconds,
          "policy_target_q": raw_target.tolist() if raw_target is not None else None,
          "handoff_alpha": handoff.alpha if handoff is not None else None,
          "root_pos": root[0].tolist() if root is not None else None,
          "root_quat": root[1].tolist() if root is not None else None,
          "root_age": now - root[3] if root is not None else None,
          "mode_machine": feedback[4],
          "kp": kp.tolist(),
          "kd": kd.tolist(),
          "imu_quat": imu_quat.tolist() if imu_quat is not None else None,
          "gyro": gyro.tolist(),
          "dds_state_age": now - feedback[-1],
          "target_q": target.tolist() if target is not None else None,
          "pd_torque_estimate": (kp * (target - q) - kd * dq).tolist()
          if target is not None
          else None,
        }
        if not result["phases"] or result["phases"][-1] != lifecycle.state:
          result["phases"].append(lifecycle.state)
          print(f"[{now - started:.2f}s] {lifecycle.state}", flush=True)
        record_row(row)
        if entry_ready:
          lifecycle.state = "STABLE"
        deadline += cfg["control_dt"]
        if (
          real_cfg["max_control_overrun"] is not None
          and clock.monotonic() - deadline > real_cfg["max_control_overrun"]
        ):
          raise TimeoutError("Startup/stable control deadline missed")
        clock.sleep(max(deadline - clock.monotonic(), 0))
        continue
      if root is None:
        raise TimeoutError("Robot mocap pose expired")
      # Manual observation: operator SELECT/B/X stops the robot. Do not
      # automatically halt on alternating velocity; other guards remain.
      pos, mocap_quat, _, _ = root
      quat = (
        imu_quat.copy()
        if root_grace is not None and root_grace.frozen and imu_quat is not None
        else root_orientation(mocap_quat, imu_quat, real_cfg)
      )
      if (
        np.degrees(np.arccos(np.clip(rotation(quat)[2, 2], -1, 1)))
        > real_cfg["max_tilt_degrees"]
      ):
        physical_limit("root_tilt", ValueError("Robot tilt exceeds configured limit"))
      i = kinematics.free_qpos
      kinematics_started = clock.monotonic()
      kinematics.data.qpos[i : i + 3], kinematics.data.qpos[i + 3 : i + 7] = pos, quat
      kinematics.data.qpos[kinematics.qpos], kinematics.data.qvel[kinematics.qvel] = (
        q,
        dq,
      )
      kinematics.forward_kinematics()
      observed = kinematics.state()
      # Optional source-rate conditioning belongs to the pose receiver, not
      # the policy timer. Existing tasks without it keep their old behavior.
      velocity_for_source = getattr(mocap.root, "velocity_for_source", None)
      if velocity_for_source is not None:
        velocity = velocity_for_source(root[2])
        if velocity is not None:
          observed.root_velocity_world = (
            np.zeros(3) if root_grace is not None and root_grace.frozen else velocity
          )
      kinematics_seconds = clock.monotonic() - kinematics_started
      observed.gyro = rotation(real_cfg["imu_to_pelvis_quaternion_wxyz"]) @ gyro
      if enter_policy:
        capture_anchor = getattr(engine, "capture_return_anchor", None)
        if capture_anchor is not None:
          # Also covers policy entry without AMP; does not overwrite an UP anchor.
          capture_anchor(observed.root_pos, now)
        if stable is not None:
          handoff = begin_handoff(now)
          engine.receiving = False
          policy_stable_since = None
          engine.start_policy(
            last_command[0], now, reset_history=True, blend_reference=False
          )
          entry_ready = False
        else:
          engine.start_policy(lifecycle.posture_target(now), now)
      estimate = mocap.ball.get(now)
      drain_observations = getattr(mocap.ball, "drain_observations", None)
      if drain_observations is not None and hasattr(engine, "update_ball_observations"):
        engine.update_ball_observations(drain_observations())
      planner_command_reader = getattr(mocap, "latest_planner_command", None)
      planner_command_updater = getattr(engine, "update_external_planner_command", None)
      if planner_command_reader is not None and planner_command_updater is not None:
        planner_packet, planner_received = planner_command_reader()
        if planner_packet is not None:
          planner_command_updater(planner_packet, planner_received)
      policy_started = clock.monotonic()
      target = engine.step(observed, estimate, now)
      policy_seconds = clock.monotonic() - policy_started
      raw_policy_target = target.copy() if target is not None else None
      kp, kd = np.asarray(cfg["robot"]["kp"]), np.asarray(cfg["robot"]["kd"])
      transition_active = handoff is not None
      if handoff is not None:
        # Keep the source balance controller running during the transfer.
        source = None
        if stable is not None:
          stable_started = clock.monotonic()
          source_target = stable.step(q, dq, gyro, imu_quat)
          if home_posture is not None:
            source_target = home_posture.command(source_target, now)
          source = (source_target, stable.kp, stable.kd)
          stable_compute_seconds = clock.monotonic() - stable_started
          stable_step_debug = getattr(stable, "debug", {})
        (target, kp, kd), done = handoff.apply((target, kp, kd), now, source=source)
        # The actuator handoff must not rewrite training's raw action history.
        engine.standing.previous = target.copy()
        engine.previous_target = target.copy()
        if engine.standing.readiness_mode == "timed":
          # HOPE-style fixed ramp + hold; measured metrics remain diagnostic.
          policy_stable_since = handoff.started + handoff.seconds if done else None
        else:
          settled, _ = engine.standing.observe(observed)
          policy_stable_since = (
            (now if policy_stable_since is None else policy_stable_since)
            if settled and done
            else None
          )
        if (
          done
          and policy_stable_since is not None
          and now - policy_stable_since >= stable_cfg.get("settle_seconds", 0.5)
        ):
          handoff = None
          engine.receiving = True
          transition_active = False
        elif (
          stable_cfg.get("handoff_timeout", 5.0) is not None
          and now - handoff.started > handoff.seconds + 10.0
        ):
          raise TimeoutError("Ping-pong handoff did not stabilize")
      command_generated_time = None
      if target is not None:
        pending_target = target.copy()
        check_command(target, q, dq)
        command_generated_time = send_command(target, kp, kd)
        last_command = (target.copy(), kp.copy(), kd.copy())
        result["commands_sent"] += 1
      row = {
        "time": now,
        "state": "TO_POLICY" if transition_active else lifecycle.state,
        "readiness_mode": engine.standing.readiness_mode,
        "oscillation": oscillation.debug,
        "phase": engine.phase,
        "policy": "pingpong",
        "policy_compute_seconds": policy_seconds,
        "kinematics_seconds": kinematics_seconds,
        "policy_target_before_handoff_q": raw_policy_target,
        "stable_policy": stable_step_debug,
        "stable_compute_seconds": stable_compute_seconds,
        "handoff_alpha": handoff.alpha if handoff is not None else None,
        "imu_gyro": gyro.tolist(),
        "anchor_pos": observed.anchor_pos.tolist(),
        "anchor_quat": observed.anchor_quat.tolist(),
        "racket_pos": observed.racket_pos.tolist(),
        "racket_normal": observed.racket_normal.tolist(),
        "root_source_stamp": root[2],
        "ball_source_stamp": getattr(estimate, "stamp", None),
        "ball_age": now - estimate.received
        if estimate is not None and hasattr(estimate, "received")
        else None,
        "io": io_diagnostics,
        "command_generated_time": command_generated_time,
        "mocap_imu_gravity_error_degrees": gravity_error,
        "control_work_seconds": clock.monotonic() - control_work_started,
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "root_age": now - root[3],
        "root_pose_frozen": bool(root_grace is not None and root_grace.frozen),
        "imu_quat": imu_quat.tolist() if imu_quat is not None else None,
        "completed": engine.completed,
        "recovered": engine.recovered,
        "q": q.tolist(),
        "dq": dq.tolist(),
        "root_pos": pos.tolist(),
        "root_quat": quat.tolist(),
        "root_quat_raw_mocap": mocap_quat.tolist(),
        "root_orientation_source": real_cfg.get("root_orientation_source", "mocap"),
        "stable_source_target_q": source[0]
        if transition_active and source is not None
        else None,
        "gyro": observed.gyro.tolist(),
        "buttons": buttons,
        "mode_machine": feedback[4],
        "dds_state_age": now - feedback[-1],
        "motion": deployment.motions[engine.motion_index].name,
        "motion_index": engine.motion_index,
        "checkpoint": str(deployment.checkpoint),
        "motor_output_enabled": True,
        "target_q": target.tolist() if target is not None else None,
        "pd_torque_estimate": (kp * (target - q) - kd * dq).tolist()
        if target is not None
        else None,
        "standing": engine.standing.debug,
        "planner": getattr(engine.planner, "diagnostic", None),
        "decision": engine.debug.get("decision"),
        "ball_position": estimate.position.tolist() if estimate else None,
        "ball_velocity": estimate.velocity.tolist() if estimate else None,
      }
      if not result["phases"] or result["phases"][-1] != row["state"]:
        result["phases"].append(row["state"])
        display_state = (
          "WAIT_BALL"
          if row["state"] == "ARMED" and engine.standing.readiness_mode == "timed"
          else row["state"]
        )
        print(
          f"[{now - started:.2f}s] {display_state} motion={engine.motion_index} "
          f"completed={engine.completed} recovered={engine.recovered}",
          flush=True,
        )
      if getattr(engine, "command_debug", None):
        row.update(
          {
            ("policy_target_q" if k == "target_q" else k): v.tolist()
            if isinstance(v, np.ndarray)
            else v
            for k, v in engine.command_debug.items()
          }
        )
      row["command_policy"] = getattr(engine, "command_policy", None)
      record_row(row)
      deadline += cfg["control_dt"]
      if (
        real_cfg["max_control_overrun"] is not None
        and clock.monotonic() - deadline > real_cfg["max_control_overrun"]
      ):
        raise TimeoutError("Policy control deadline missed")
      clock.sleep(max(deadline - clock.monotonic(), 0))
  except KeyboardInterrupt:
    result["termination"] = "interrupted"
  except SimulationReset:
    if simulation_io is None:
      raise
    result["termination"] = "simulation_reset"
    raise
  except Exception as exc:
    result["termination"] = "error"
    result["failure"] = {
      "phase": "TO_POLICY"
      if handoff is not None and lifecycle.state == "ARMED"
      else lifecycle.state,
      "motion_index": engine.motion_index,
      "error": str(exc),
    }
    fault_snapshot = {
      "time": clock.monotonic(),
      "wall_time": clock.time(),
      "failure": result["failure"].copy(),
      "last_trace_row": json.loads(last_row) if isinstance(last_row, str) else last_row,
      "last_available_feedback": feedback,
      "attempted_target_q": pending_target,
      "last_sent_command": last_command,
      "standing_policy": engine.standing.controller.debug,
      "swing_policy": engine.controller.debug,
      "stable_policy": getattr(stable, "debug", None),
      "oscillation": oscillation.debug,
      "io": robot.diagnostics()
      if robot is not None and hasattr(robot, "diagnostics")
      else None,
    }
    lifecycle.halt()
    if (
      real_cfg.get("latch_damping_on_fault", False)
      and enable_motors
      and robot is not None
      and feedback is not None
    ):
      from deploy.g1_pingpong.hope_safety import wait_in_damping

      print(f"[HALT] {exc}; latched damping, SELECT/B/X or Ctrl-C to exit", flush=True)
      result["safety_halt"] = {"reason": str(exc), "latched": True}

      def record_halt(packet, error):
        nonlocal cycle_index, previous_cycle, loop_period, loop_lateness
        now = clock.monotonic()
        cycle_index += 1
        loop_period = now - previous_cycle if previous_cycle is not None else None
        previous_cycle, loop_lateness = now, 0.0
        feedback_context.clear()
        if packet is not None and len(packet) == 7:
          feedback_context["feedback_callback_time"] = packet[-1]
        row = {
          "time": now,
          "state": "HALT",
          "policy": "damping",
          "failure": result["failure"],
          "io_error": error,
          "io": robot.diagnostics(),
          "command_generated_time": None,
        }
        if packet is not None:
          row.update(q=packet[0].tolist(), dq=packet[1].tolist(), buttons=packet[3])
        record_row(row)

      try:
        result["termination"] = wait_in_damping(
          robot, clock, started, duration, record_halt
        )
      except KeyboardInterrupt:
        result["termination"] = "interrupted"
    else:
      raise
  finally:
    try:
      if robot:
        robot.close()
    finally:
      try:
        if mocap:
          mocap.close()
      finally:
        if recorder:
          recorder.close()
          result["trace_recording"] = recorder.diagnostics()
        if robot is not None and hasattr(robot, "diagnostics"):
          result["final_io_diagnostics"] = robot.diagnostics()
        if mocap is not None and hasattr(mocap, "diagnostics"):
          result["final_mocap_diagnostics"] = mocap.diagnostics()
        result.update(
          state=lifecycle.state if enable_motors else "READ_ONLY",
          phase=engine.phase if enable_motors else "READ_ONLY",
          completed_swings=engine.completed,
          recovered_swings=engine.recovered,
          elapsed_seconds=clock.monotonic() - started,
        )
        if summary_path:
          if fault_snapshot is not None:
            snapshot_path = summary_path.with_name(summary_path.stem + ".fault.json")
            snapshot_path.write_text(
              json.dumps(fault_snapshot, default=json_default, indent=2)
            )
            result["fault_snapshot"] = str(snapshot_path)
          summary_path.write_text(json.dumps(result, default=json_default, indent=2))
          if (
            real_cfg.get("latency_auto_summary", False)
            and summary_path.name == "summary.json"
          ):
            try:
              from deploy.tools.analyze_latency import summarize

              report_path = summary_path.parent / "latency_summary.json"
              report_path.write_text(
                json.dumps(summarize(summary_path.parent), indent=2)
              )
              result["latency_analysis"] = {"path": str(report_path), "error": None}
            except Exception as analysis_error:
              # Diagnostics must not replace the original fault or stop cleanup.
              result["latency_analysis"] = {"error": str(analysis_error)}
            summary_path.write_text(json.dumps(result, default=json_default, indent=2))
  return result

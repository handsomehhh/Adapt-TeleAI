"""Bounded diagnostic writers; disk IO never runs in a control callback."""

import json
import queue
import threading
from pathlib import Path

import numpy as np


def json_default(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  raise TypeError(f"Cannot record {type(value).__name__}")


class BufferedWriter:
  def __init__(self, path, *, dtype=None, capacity=2048, prefix="lowstate"):
    self.path = Path(path)
    self.dtype = dtype
    self.prefix = prefix
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is not None:
      self.path.mkdir(parents=True, exist_ok=True)
    self.queue = queue.Queue(maxsize=capacity)
    self.stop = threading.Event()
    self.accepted = self.written = self.dropped = self.chunks = 0
    self.error = None
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def write(self, item):
    if self.stop.is_set() or self.error is not None:
      self.dropped += 1
      return
    try:
      self.queue.put_nowait(item)
      self.accepted += 1
    except queue.Full:
      self.dropped += 1

  def _chunk(self, rows):
    np.save(
      self.path / f"{self.prefix}_{self.chunks:06d}.npy",
      np.array(rows, dtype=self.dtype),
    )
    self.written += len(rows)
    self.chunks += 1

  def _run(self):
    stream = None
    rows = []
    try:
      if self.dtype is None:
        stream = self.path.open("w")
      while not self.stop.is_set() or not self.queue.empty():
        try:
          item = self.queue.get(timeout=0.1)
        except queue.Empty:
          continue
        if stream is not None:
          stream.write(item)
          self.written += 1
          if self.written % 50 == 0:
            stream.flush()
        else:
          rows.append(item)
          if len(rows) >= 256:
            self._chunk(rows)
            rows = []
      if rows:
        self._chunk(rows)
    except Exception as exc:
      self.error = str(exc)
    finally:
      if stream is not None:
        stream.close()

  def diagnostics(self):
    return dict(
      accepted=self.accepted,
      written=self.written,
      dropped=self.dropped,
      queue_size=self.queue.qsize(),
      error=self.error,
      writer_alive=self.thread.is_alive(),
      chunks=self.chunks,
    )

  def close(self):
    self.stop.set()
    self.thread.join(timeout=5)


LOWSTATE_DTYPE = np.dtype(
  [
    ("callback_time", "<f8"),
    ("source_tick", "<u4"),
    ("mode_machine", "u1"),
    ("buttons", "<u2"),
    ("q", "<f4", (29,)),
    ("dq", "<f4", (29,)),
    ("tau_est", "<f4", (29,)),
    ("gyro", "<f4", (3,)),
    ("imu_quat_wxyz", "<f4", (4,)),
    ("motorstate", "<u4", (29,)),
    ("voltage", "<f4", (29,)),
    ("temperature", "<i2", (29, 2)),
    ("last_written_origin", "<f8"),
    ("writer_halted", "u1"),
    ("callback_sequence", "<u8"),
    ("last_write_sequence", "<u8"),
    ("last_written_command_id", "<u8"),
    ("last_write_started_time", "<f8"),
    ("last_write_completed_time", "<f8"),
  ]
)


class LowStateRecorder(BufferedWriter):
  def __init__(self, path, joint_names, hz):
    if not np.isfinite(hz) or not 0 < hz <= 1000:
      raise ValueError("record_lowstate_hz must be in (0, 1000]")
    self.interval = 1.0 / hz
    self.next_sample = None
    super().__init__(path, dtype=LOWSTATE_DTYPE)
    (self.path / "schema.json").write_text(
      json.dumps(
        {
          "joint_names": joint_names,
          "requested_hz": hz,
          "dtype": LOWSTATE_DTYPE.descr,
          "clock": "host monotonic; source_tick is unconverted device counter",
          "schema_version": 2,
          "last_written_origin": "host command generation time; Write is not actuation ACK",
        },
        indent=2,
      )
    )

  def due(self, now):
    if self.next_sample is None:
      self.next_sample = now
    if now < self.next_sample:
      return False
    self.next_sample += (
      int((now - self.next_sample) / self.interval) + 1
    ) * self.interval
    return True


class FeedbackTiming:
  """Observe every SDK callback; device tick is not a synchronized clock."""

  def __init__(self):
    self.sequence = 0
    self.previous_time = self.previous_tick = None
    self.duplicates = self.tick_discontinuities = 0
    self.max_interval = self.max_interval_time = 0.0

  def observe(self, now, tick):
    interval = None if self.previous_time is None else now - self.previous_time
    delta = None if self.previous_tick is None else (tick - self.previous_tick) % 2**32
    self.sequence += 1
    if delta == 0:
      self.duplicates += 1
    if delta is not None and delta > 2**31:
      self.tick_discontinuities += 1
    if interval is not None and interval > self.max_interval:
      self.max_interval, self.max_interval_time = interval, now
    self.previous_time, self.previous_tick = now, tick
    return dict(
      feedback_sequence=self.sequence,
      callback_interval_seconds=interval,
      source_tick_delta=delta,
      callback_interval_max_seconds=self.max_interval,
      callback_interval_max_time=self.max_interval_time,
      duplicate_source_ticks=self.duplicates,
      source_tick_discontinuities=self.tick_discontinuities,
    )


COMMAND_WRITE_DTYPE = np.dtype(
  [
    (name, "<f8")
    for name in (
      "generated_time",
      "accepted_time",
      "feedback_callback_time",
      "feedback_consumed_time",
      "feedback_packet_enqueued_time",
      "feedback_packet_dequeued_time",
      "prepare_started_time",
      "scheduled_time",
      "write_started_time",
      "write_completed_time",
      "writer_feedback_callback_time",
      "write_interval_seconds",
    )
  ]
  + [
    (name, "<u8")
    for name in (
      "write_sequence",
      "command_id",
      "control_cycle",
      "feedback_sequence",
    )
  ]
  + [
    ("source_tick", "<i8"),
    ("writer_source_tick", "<i8"),
    ("first_write", "u1"),
    ("halted", "u1"),
    ("write_result", "i1"),
    ("q_target", "<f4", (29,)),
    ("kp", "<f4", (29,)),
    ("kd", "<f4", (29,)),
    ("command_dequeued_time", "<f8"),
  ]
)


class CommandWriteRecorder(BufferedWriter):
  def __init__(self, path, joint_names):
    super().__init__(path, dtype=COMMAND_WRITE_DTYPE, capacity=8192, prefix="commands")
    (self.path / "schema.json").write_text(
      json.dumps(
        {
          "schema_version": 1,
          "joint_names": joint_names,
          "dtype": COMMAND_WRITE_DTYPE.descr,
          "clock": "host monotonic seconds across processes on this host",
          "sampling": "every actual DDS Write, including repeated and damping commands",
          "command_id": "policy-process sequence; repeated writes retain this ID",
          "write_sequence": "IO-process sequence for each DDS Write attempt",
          "source_tick": "unconverted device counter associated with policy feedback",
          "wire_contract": "mode_pr=0, dq_target=0, tau_feedforward=0; recorded q/kp/kd in policy joint order",
          "limitations": "Write completion is not robot receipt or motor actuation acknowledgement; closed-loop response lag is not an independent one-way delay measurement",
        },
        indent=2,
      )
    )

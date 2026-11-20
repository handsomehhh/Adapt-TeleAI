"""Bounded live history with append-only disk chunks for uninterrupted tests."""

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def atomic_output(path):
  """Only publish a fully closed file; interruption preserves an older snapshot."""
  path = Path(path)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(
      dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
      temporary = Path(stream.name)
      yield stream
    temporary.replace(path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def write_summary(path, summary):
  with atomic_output(path) as stream:
    stream.write((json.dumps(summary, indent=2) + "\n").encode())


def write_trace(path, rows):
  if rows:
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    with atomic_output(path) as stream:
      np.savez_compressed(stream, allow_pickle=False, **arrays)


class TraceBuffer(list):
  def __init__(self):
    super().__init__()
    self.total = 0
    self.directory = None
    self.chunk_steps = 1000
    self.parts = []
    self.just_flushed = False
    self.ever_fallen = False
    self.sums = dict.fromkeys(("joint_rms_rad", "racket_error_m", "fallen"), 0.0)

  def append(self, row):
    super().append(row)
    self.total += 1
    self.ever_fallen |= row["fallen"]
    for key in self.sums:
      self.sums[key] += row[key]
    self.just_flushed = False
    if self.directory is not None and len(self) >= 2 * self.chunk_steps:
      name = f"chunks/trace_{len(self.parts):06d}.npz"
      path = self.directory / name
      path.parent.mkdir(parents=True, exist_ok=True)
      write_trace(path, self[: self.chunk_steps])
      self.parts.append(name)
      del self[: self.chunk_steps]
      self.just_flushed = True

  def write(self, output):
    output = Path(output).resolve()
    if self.parts and output != self.directory:
      (output / "chunks").mkdir(exist_ok=True)
      for part in self.parts:
        shutil.copy2(self.directory / part, output / part)
    write_trace(output / "trace.npz", self)

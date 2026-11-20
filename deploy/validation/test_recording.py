"""An interrupted save must not corrupt an already published recording."""

import json

import numpy as np
import pytest

from deploy.recording import write_summary, write_trace


def test_interrupted_trace_write_preserves_previous_snapshot(tmp_path, monkeypatch):
  path = tmp_path / "trace.npz"
  write_trace(path, [{"time": 0.02, "qpos": [0.0, 1.0]}])
  original = path.read_bytes()

  def interrupted(stream, **_kwargs):
    stream.write(b"incomplete compressed archive")
    raise KeyboardInterrupt

  monkeypatch.setattr("deploy.recording.np.savez_compressed", interrupted)
  with pytest.raises(KeyboardInterrupt):
    write_trace(path, [{"time": 0.04, "qpos": [2.0, 3.0]}])
  assert path.read_bytes() == original
  with np.load(path, allow_pickle=False) as archive:
    assert archive["time"].tolist() == [0.02]
  assert not list(tmp_path.glob("*.tmp"))


def test_interrupted_metadata_write_preserves_previous_summary(tmp_path, monkeypatch):
  path = tmp_path / "summary.json"
  write_summary(path, {"steps": 0, "completed": False})

  def interrupted(*_args, **_kwargs):
    raise KeyboardInterrupt

  monkeypatch.setattr("deploy.recording.json.dumps", interrupted)
  with pytest.raises(KeyboardInterrupt):
    write_summary(path, {"steps": 99, "completed": True})
  assert json.loads(path.read_text()) == {"steps": 0, "completed": False}
  assert not list(tmp_path.glob("*.tmp"))

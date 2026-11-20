"""Replay recorded states verbatim; no policies, physics stepping or rescoring."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from deploy.world import World
from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import G1_PINGPONG_XML
from mjlab.tasks.g1_adapt.motion import G1Kinematics
from mjlab.tasks.g1_adapt.mvae import file_hash


class Replay:
  def __init__(self, trace):
    self.trace = Path(trace).resolve()
    self.error: str | None = None
    self.summary = json.loads(self.trace.with_name("summary.json").read_text())
    if self.summary["asset_sha256"] != file_hash(G1_PINGPONG_XML):
      raise ValueError("Replay robot asset differs from the recording")
    parts = []
    for name in [*self.summary.get("trace_chunks", []), self.trace.name]:
      with np.load(self.trace.parent / name, allow_pickle=False) as archive:
        parts.append({key: archive[key] for key in archive.files})
    self.arrays = {
      key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    count = len(self.arrays["qpos"])
    if count < 1 or any(len(value) != count for value in self.arrays.values()):
      raise ValueError("Empty or inconsistent replay arrays")
    self.records = [
      {key: value[i].tolist() for key, value in self.arrays.items()}
      for i in range(count)
    ]
    self.world = World()
    self.mode, self.scenario = "replay", "replay"
    self.trials = self.summary["trials"]
    self.latent, self.speed = np.zeros(32), 1.0
    self.controller = SimpleNamespace(current=SimpleNamespace(fk=G1Kinematics("cpu")))
    self.ball = SimpleNamespace(p=torch.zeros(1, 3))
    self.ball_obs, self.future = torch.zeros(1, 101), torch.zeros(1, 30, 3)
    self.reset()

  def reset(self):
    self.paused = True
    self.seek(0)

  def seek(self, index):
    self.index = min(max(int(index), 0), len(self.records) - 1)
    row = self.records[self.index]
    self.world.data.qpos[:] = row["qpos"]
    self.world.data.qvel[:] = row["qvel"]
    self.world.data.ctrl[self.world.actuators] = row["targets"]
    self.world.refresh()
    self.info, self.elapsed, self.active = row, row["time"], row["ball_active"]
    self.ball.p = torch.tensor([row["ball"]], dtype=torch.float32)
    reference = torch.tensor([row["reference_qpos"]], dtype=torch.float32)
    current = self.controller.current
    current.positions, current.quaternions = current.fk(
      reference[:, :3], reference[:, 3:7], reference[:, 7:]
    )
    if row["planner_obs_valid"]:
      self.ball_obs = torch.tensor([row["planner_obs"][-101:]])
      self.future = (
        self.ball_obs[:, 6:96].reshape(1, 30, 3)
        + self.world.robot_state.root_position[:, None]
      )
    else:
      self.ball_obs.zero_()
      self.future.zero_()
    return row

  def step(self):
    if self.index + 1 >= len(self.records):
      self.paused = True
      return self.info
    return self.seek(self.index + 1)

  def set_mode(self, mode):
    if mode != "replay":
      raise ValueError("Start a new session to leave replay mode")

  def feed(self):
    return False

  def save(self, directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != self.trace.parent:
      shutil.copy2(self.trace, output / "trace.npz")
      shutil.copy2(self.trace.with_name("summary.json"), output / "summary.json")
      for name in self.summary.get("trace_chunks", []):
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.trace.parent / name, output / name)
    return self.summary

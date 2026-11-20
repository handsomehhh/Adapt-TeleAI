"""Wall-clock pacing and display-only interpolation; policy steps stay fixed."""

import time
from typing import Any

import mujoco
import numpy as np


class PlaybackClock:
  def __init__(self, dt=0.02, max_catchup=5):
    self.dt, self.max_catchup = dt, max_catchup
    self.last = None
    self.accumulator = 0.0
    self.rate = 1.0
    self.was_paused = True
    self.dropped = 0.0

  def advance(self, now, paused, rate):
    elapsed = 0.0 if self.last is None else max(0.0, now - self.last)
    self.last = now
    if paused or self.was_paused or rate != self.rate:
      elapsed = 0.0
      self.accumulator = 0.0
    self.was_paused, self.rate = paused, rate
    self.accumulator += elapsed * rate
    limit = self.max_catchup * self.dt
    self.dropped += max(0.0, self.accumulator - limit)
    self.accumulator = min(self.accumulator, limit)
    count = int((self.accumulator + 1e-10) / self.dt)
    self.accumulator = max(0.0, self.accumulator - count * self.dt)
    return count

  @property
  def fraction(self):
    return min(1.0, self.accumulator / self.dt)


def interpolate_qpos(before, after, fraction):
  """Interpolate joint positions and the free-root quaternion for rendering."""
  if fraction <= 0:
    return before.copy()
  if fraction >= 1:
    return after.copy()
  qpos = (1 - fraction) * before + fraction * after
  first, last = before[3:7], after[3:7]
  if np.dot(first, last) < 0:
    last = -last
  quat = (1 - fraction) * first + fraction * last
  qpos[3:7] = quat / max(np.linalg.norm(quat), 1e-12)
  return qpos


class DisplayFrames:
  """Copies snapshots; interpolated state never enters the physics/controller."""

  def __init__(self, session):
    self.data = mujoco.MjData(session.world.model)
    self.model = session.world.model
    self.before = self.after = None
    self.capture(session)

  def capture(self, session):
    row = session.info
    token = (session.mode, row.get("segment", 0), row["time"], session.active)
    if self.after is not None and token == self.after["token"]:
      return
    frame: dict[str, Any] = {
      key: np.array(row[key], dtype=float)
      for key in (
        "qpos",
        "reference_qpos",
        "ball",
        "racket",
        "reference_racket",
        "normal",
      )
    }
    frame["positions"] = session.controller.current.positions[0].numpy().copy()
    frame["ball"] = session.ball.p[0].numpy().copy()
    frame["future"] = session.future[0].numpy().copy()
    frame["token"] = token
    previous = self.after
    continuous = (
      previous is not None
      and previous["token"][:2] == token[:2]
      and previous["token"][3] == token[3]
      and 0 < token[2] - previous["token"][2] <= session.world.control_dt * 1.01
    )
    self.before = previous if continuous else frame
    self.after = frame

  def sample(self, fraction=1.0):
    assert self.before is not None and self.after is not None
    fraction = float(np.clip(fraction, 0, 1))
    frame = {
      key: (1 - fraction) * self.before[key] + fraction * self.after[key]
      for key in ("ball", "racket", "reference_racket", "normal", "positions", "future")
    }
    for key in ("qpos", "reference_qpos"):
      frame[key] = interpolate_qpos(self.before[key], self.after[key], fraction)
    frame["normal"] /= max(np.linalg.norm(frame["normal"]), 1e-12)
    self.data.qpos[:] = frame["qpos"]
    mujoco.mj_kinematics(self.model, self.data)
    return frame


def run_viewer(viewer, directory, max_steps=0):
  """Run fixed control steps and a separate 60 Hz display on one state thread."""
  session = viewer.session
  clock = PlaybackClock(session.world.control_dt)
  next_frame = time.monotonic()
  window_start, frames, window_steps, steps = next_frame, 0, 0, 0
  while viewer.running and (max_steps == 0 or steps < max_steps):
    viewer.drain(directory)
    if not viewer.running:
      break
    viewer.frames.capture(session)
    now = time.monotonic()
    count = clock.advance(now, session.paused, viewer.rate)
    if viewer.single_step:
      count = 1
      clock.accumulator = 0.0
    for _ in range(count):
      if not viewer.running or (max_steps and steps >= max_steps):
        break
      session.step()
      viewer.frames.capture(session)
      steps += 1
      window_steps += 1
      if session.paused:
        break
    if now >= next_frame:
      viewer.update(1.0 if session.paused else clock.fraction)
      frames += 1
      next_frame = now + 1 / 60
    span = time.monotonic() - window_start
    if span >= 1:
      viewer.playback_stats = {
        "fps": frames / span,
        "realtime_factor": window_steps * session.world.control_dt / span,
        "dropped_s": clock.dropped,
      }
      window_start, frames, window_steps = time.monotonic(), 0, 0
    time.sleep(min(0.003, max(0.0005, next_frame - time.monotonic())))

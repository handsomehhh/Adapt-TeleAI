"""MuJoCo trial lifecycle and measurement; agent is a replaceable adapter.

Required adapter: robot.model/data, dt, translation, reset(), advance(command),
measure(), lost_standing(sample), accepted, decision, metadata. This module
never accesses motion libraries, actor weights or selector internals.
"""

import json
import threading
import time

import mujoco
import numpy as np

from deploy.backends.playback import HELP as PLAYBACK_HELP
from deploy.backends.playback import PlaybackControl
from deploy.benchmarks.racket_command.domain import (
  contains,
  grid_edges,
  supports_vectors,
)
from deploy.benchmarks.racket_command.interfaces import CommandAgent
from deploy.benchmarks.racket_command.metrics import evaluate
from deploy.benchmarks.racket_command.visualization import draw_scene


class MujocoCommandEnvironment:
  def __init__(
    self,
    agent: CommandAgent,
    config,
    output,
    render=False,
    playback_rate=1,
    capture_first=0,
    show_ball=True,
    command_only=False,
  ):
    self.agent, self.config, self.output = agent, config, output
    self.viewer = None
    self.renderer = None
    self.playback_rate = playback_rate
    self.playback = PlaybackControl(playback_rate)
    self.capture_first = capture_first
    self.show_ball = show_ball
    self.command_only = command_only
    self.ball = None
    self.completed = []
    self.paused = self.quit = False
    self.viewer_threads = ()
    if render:
      from deploy import viewer

      before = set(threading.enumerate())
      self.viewer = viewer.launch_passive(
        agent.robot.model, agent.robot.data, key_callback=self.key
      )
      self.viewer_threads = (
        ()
        if hasattr(self.viewer, "server")
        else tuple(t for t in threading.enumerate() if t not in before)
      )
      with self.viewer.lock():
        self.viewer.cam.lookat[:] = [1.3, 0, 0.9]
        self.viewer.cam.distance = 4.6
        self.viewer.cam.azimuth = 125
        self.viewer.cam.elevation = -20
      print(f"MuJoCo playback: {PLAYBACK_HELP}", flush=True)
    if capture_first:
      self.renderer = mujoco.Renderer(agent.robot.model, height=720, width=960)

  def key(self, key):
    if self.playback.key(key):
      self.playback_rate = self.playback.rate
      return
    if key in (ord(" "), ord("P")):
      self.paused = not self.paused
    if key in (ord("Q"), 256):
      self.quit = True

  def display(self, trial, samples, decision=None):
    if self.viewer:
      if not self.viewer.is_running():
        self.quit = True
        return
      with self.viewer.lock():
        draw_scene(
          self.viewer.user_scn,
          self.config,
          trial,
          samples,
          self.agent.translation,
          self.completed,
          show_ball=self.show_ball,
          ball=self.ball,
        )
      t = samples[-1]["time"] if samples else 0
      last = (
        self.completed[-1]
        if self.completed and self.completed[-1]["trial_id"] == trial.trial_id
        else None
      )
      error_text = (
        f"p={last.get('position_error_m', 0):.3f}m, "
        f"n={last.get('normal_error_degrees', 0):.1f}deg, "
        f"v={last.get('velocity_error_mps', 0):.2f}m/s, success={last['success']}"
        if last
        else "pending"
      )
      ball_status = (
        (
          f"contact={self.ball.contact is not None}, net_hit={self.ball.net_hit}, "
          f"speed={np.linalg.norm(self.ball.velocity):.2f}m/s"
        )
        if self.ball
        else "No incoming ball / command-only"
      )
      point_text = (
        f"{trial.point_id} / direction {trial.direction_id} / "
        f"p={np.round(trial.position, 3).tolist()} / speed={np.linalg.norm(trial.velocity):.2f}"
        if trial.point_id is not None
        else "random position"
      )
      ys, zs = grid_edges(self.config)
      grid_text = (
        f"{self.config['grid'][0]}Y x {self.config['grid'][1]}Z; "
        f"X={self.config['plane_x']:.3f}m"
      )
      bounds_text = (
        f"Y=[{ys[trial.cell_y]:.3f}, {ys[trial.cell_y + 1]:.3f}], "
        f"Z=[{zs[trial.cell_z]:.3f}, {zs[trial.cell_z + 1]:.3f}] m"
      )
      self.viewer.set_texts(
        (
          None,
          None,
          "Independent racket commands\nGrid / plane\nTrial / cell Y,Z (zero-based)\nCurrent cell bounds\nFixed point / direction\nTTS / countdown\nSelector\nDeadline result\n"
          + ("" if self.command_only else "Ball dynamics\n")
          + "Gold=target, magenta=velocity\nCyan=measured, red=deadline\n"
          + ("" if self.command_only else "Orange=physical ball\n")
          + "Yellow border=current cell\nSpace=pause, Q=quit\nPlayback [ / ] / \\",
          (
            "Stage 1: policy tracking; no ball/planner\n"
            if self.command_only
            else "Command tracking + physical contact\n"
          )
          + f"{grid_text}\n{trial.trial_id} / {trial.cell_y},{trial.cell_z}\n{bounds_text}\n"
          f"{point_text}\n"
          f"{trial.tts:.3f}s / {trial.tts - t:+.3f}s\n{(decision or {}).get('reason', 'pending')}\n{error_text}"
          + ("" if self.command_only else f"\n{ball_status}")
          + f"\n{self.playback.rate:g}x",
        )
      )
      self.viewer.sync()

  def capture(self, trial, samples, tag="deadline"):
    import matplotlib.image as mpimg

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [1.3, 0, 0.9]
    camera.distance, camera.azimuth, camera.elevation = 4.6, 125, -20
    self.renderer.update_scene(self.agent.robot.data, camera=camera)
    # Offscreen scene includes physical robot/table; append overlay geometry.
    overlay = mujoco.MjvScene(self.agent.robot.model, maxgeom=1000)
    draw_scene(
      overlay,
      self.config,
      trial,
      samples,
      self.agent.translation,
      self.completed,
      show_ball=self.show_ball,
      ball=self.ball,
    )
    scene = self.renderer.scene
    for i in range(overlay.ngeom):
      if scene.ngeom >= scene.maxgeom:
        break
      source, target = overlay.geoms[i], scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        target, source.type, source.size, source.pos, source.mat.ravel(), source.rgba
      )
      # MjvGeom connector sizes differ from initGeom's input dimensions.
      target.size[:] = source.size
      target.emission = source.emission
      target.label = source.label
      scene.ngeom += 1
    folder = self.output / "frames"
    folder.mkdir(exist_ok=True)
    mpimg.imsave(
      folder / f"trial_{trial.trial_id if trial is not None else 0:05d}_{tag}.png",
      self.renderer.render(),
    )

  def run_trial(self, trial, ordinal):
    self.agent.reset()
    if not self.command_only:
      from deploy.benchmarks.racket_command.physical_ball import make_incoming
    self.ball = (
      None
      if self.command_only
      else make_incoming(trial, self.config["seed"], self.config.get("physical_ball"))
    )
    samples = [self.agent.measure()]
    if self.ball:
      samples[0]["ball_position"] = self.ball.position.tolist()
    self.playback.reset(samples[0]["time"])
    if self.ball:
      self.ball.previous_racket = np.asarray(samples[0]["position"])
    failure = None
    accepted = False
    decision = None
    motion = None
    captured = False
    captured_events = set()
    while samples[-1]["time"] < self.config["episode_seconds"]:
      while self.paused and not self.quit:
        self.display(trial, samples, decision)
        time.sleep(0.02)
      if self.quit:
        failure = "interrupted"
        break
      wall = time.monotonic()
      try:
        self.agent.advance(
          trial.command(0) if len(samples) == 1 else None,
          on_physics_step=self.ball.step if self.ball else None,
        )
        if len(samples) == 1:
          accepted, decision = self.agent.accepted, self.agent.decision
          motion = self.agent.measure()["motion_index"] if accepted else None
        sample = self.agent.measure()
        if self.ball:
          sample["ball_position"] = self.ball.position.tolist()
        samples.append(sample)
        if self.agent.lost_standing(sample):
          failure = "lost_standing"
          break
      except (ValueError, RuntimeError, TimeoutError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        break
      self.display(trial, samples, decision)
      if (
        self.renderer
        and ordinal < self.capture_first
        and not captured
        and sample["time"] >= trial.tts
      ):
        self.capture(trial, samples)
        captured = True
      if self.renderer and ordinal < self.capture_first and self.ball:
        for tag, event in [
          ("contact", self.ball.contact),
          ("landing", self.ball.first_landing),
        ]:
          if event is not None and tag not in captured_events:
            self.capture(trial, samples, tag)
            captured_events.add(tag)
      if self.viewer:
        self.playback.wait(sample["time"], frame_started=wall)
      if (
        self.config.get("rejected_deadline_only")
        and not accepted
        and sample["time"] >= trial.tts + 0.25
      ):
        break  # Already a failed attempt; retain the full deadline diagnostic window.
    recovered = bool(accepted and samples[-1].get("recovered"))
    result = evaluate(
      trial, samples, accepted, recovered, failure, self.config["tolerances"]
    )
    ball_report = (
      self.ball.report()
      if self.ball
      else dict(
        available=False,
        reason="command_only"
        if self.command_only
        else "no_legal_incoming_for_sampled_command",
      )
    )
    result.update(
      decision=decision,
      motion_index=motion,
      max_tilt_degrees=max(s["tilt_degrees"] for s in samples),
    )
    domain = self.config.get("coverage_domain")
    if domain:
      members = [
        region["source_motion"]
        for region in domain["regions"]
        if contains(region, np.asarray(trial.position))
        and supports_vectors(
          region, np.asarray(trial.normal), np.asarray(trial.velocity)
        )
      ]
      result["command_supported_motions"] = members
      result["command_in_training_support"] = bool(members)
      result["selected_motion_exported_support"] = (
        motion in members if accepted else None
      )
    if not self.command_only:
      if self.ball and self.ball.contact:
        result["racket_ball_distance_at_contact_m"] = self.ball.contact[
          "racket_ball_distance_m"
        ]
      result.update(
        ball=ball_report,
        physical_feed_available=self.ball is not None,
        ball_hit=self.ball.contact is not None if self.ball else False,
        legal_return=ball_report.get("legal_return", False),
      )
    if not self.config.get("retain_ball_trajectory", True):
      ball_report.pop("trajectory", None)
    self.completed.append(result)
    traces = self.output / "traces"
    traces.mkdir(exist_ok=True)
    if self.config.get("retain_traces", True):
      (traces / f"trial_{trial.trial_id:05d}.json").write_text(json.dumps(samples))
    if self.viewer:
      until = time.monotonic() + self.config["display_hold_seconds"]
      while time.monotonic() < until and not self.quit:
        self.display(trial, samples, decision)
        time.sleep(0.02)
    return result

  def close(self):
    if self.renderer:
      self.renderer.close()
    if self.viewer:
      self.viewer.close()
      for thread in self.viewer_threads:
        thread.join(timeout=5)

"""Measured incoming ball -> position sensor -> A3 -> selector -> G1 actor.

The incoming plant follows recorded positions on a reconstructed180Hz physical
clock until racket contact or the last uncontaminated incoming frame. It then
uses the unchanged deployment SimBall dynamics. The planner never sees the
trajectory, future samples, physical velocities, or recorded racket commands.
All custom IO is scoped to this benchmark; production runtime is not modified.
"""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import yaml

from deploy.backends.mujoco import SimBall
from deploy.g1_pingpong.ball_physics import swept_racket_collision
from deploy.g1_pingpong.simulation_runtime import MujocoRuntime, SimulatedMocap, run_deployment
from deploy.hope_g1.configuration import load_deployment
from deploy.hope_g1.controller import HopeController

DEFAULT_INPUT = Path(__file__).with_name("recorded_ball_cases_20261022.json")


def capture_grid(headers, hz=180.):
  """Infer frame-grid gaps from100Hz callback batches without changing positions.

  Acquisition stamps are absent. Batches (<1ms separation) contain one or more
  frames; their final grid index is inferred from callback time. Strictly
  increasing indices retain missing-frame gaps instead of compressing flight.
  This is a playback-time reconstruction, not proof of true acquisition epochs.
  """
  headers = np.asarray(headers)
  starts = np.r_[0, np.flatnonzero(np.diff(headers) >= .001)+1]
  ends = np.r_[starts[1:], len(headers)]
  indices = np.zeros(len(headers), dtype=int)
  previous = -1
  for start, end in zip(starts, ends, strict=True):
    last = max(int(np.floor(headers[end-1]*hz+1e-8)), previous+end-start)
    indices[start:end] = np.arange(last-(end-start)+1, last+1)
    previous = last
  return (indices-indices[0])/hz


class RecordedBall(SimBall):
  @classmethod
  def create(cls, cfg, case, translation, mode):
    samples = np.asarray(case["samples"], dtype=float)
    capture = capture_grid(samples[:, 0])
    # A single epoch translation keeps all reconstructed acquisition times
    # causal relative to their original callback/arrival times.
    shift = max(0., float(np.max(capture-samples[:, 0])))
    points = samples[:, 2:5]-translation
    velocity = (points[1]-points[0])/(capture[1]-capture[0])
    ball = cls.from_launch(cfg, points[0], velocity, lifetime=3.)
    ball.case, ball.mode, ball.translation = case, mode, translation
    ball.capture_times, ball.recorded_positions = capture, points
    ball.sensor_times = samples[:, :2]+shift
    ball.reconstructed_clock_shift = shift
    ball.next_frame, ball.released, ball.release_reason = 0, False, None
    ball.history = [[0., *points[0].tolist()]]
    ball.spawn_time = None
    ball.last_sensor_stamp = -1.
    ball.feed_metadata = dict(mode="recorded_incoming", case=case["id"],
      sensor_clock=mode, future_samples_given_to_planner=False,
      measured_frames=len(samples), raw_source_sha256=case["source_packet_sha256"],
      capture_clock="reconstructed180Hz frame grid with inferred missing-frame gaps; source acquisition stamps absent",
      epoch_shift_s=shift, excluded="recorded real-robot deflections after incoming cutoff")
    return ball

  def substep(self, state, dt):
    if self.released or self.elapsed+dt > self.capture_times[-1]+1e-10:
      if not self.released:
        self.released, self.release_reason = True, "end_of_uncontaminated_recording"
      super().substep(state, dt)
    else:
      p0 = self.position.copy()
      t = self.elapsed+dt
      p1 = np.array([np.interp(t, self.capture_times, self.recorded_positions[:, i])
                     for i in range(3)])
      v = (p1-p0)/dt
      previous = state.racket_pos if self.previous_racket is None else self.previous_racket
      tensor = lambda value: torch.as_tensor(value, dtype=torch.float32).reshape(1, 3)
      p2, v2, hit = swept_racket_collision(
        tensor(p0), tensor(p1), tensor(v), tensor(previous), tensor(state.racket_pos),
        tensor(state.racket_normal), dt, ball_radius=self.cfg["ball_radius"],
        racket_radius=self.cfg["racket_radius"], restitution=self.cfg["racket_restitution"])
      self.elapsed = t
      self.position, self.velocity = p1, v
      if bool(hit.item()):
        self.hit = self.released = True
        self.release_reason = "mujoco_racket_contact"
        self.position, self.velocity = p2.numpy()[0].copy(), v2.numpy()[0].copy()
        self.contact = dict(time=t, position=self.position.tolist(), incoming_velocity=v.tolist(),
          outgoing_velocity=self.velocity.tolist(), racket_velocity=((state.racket_pos-previous)/dt).tolist(),
          racket_normal=state.racket_normal.tolist())
      else:
        # Keep exact acquisition knots in the physical history. Re-interpolating
        # only the1ms integration grid across a bounce would smooth a real
        # measured frame and change the six-sample velocity fit.
        knots = np.flatnonzero((self.capture_times > t-dt+1e-10)
                               & (self.capture_times < t-1e-10))
        for knot in knots:
          self.history.append([float(self.capture_times[knot]),
                               *self.recorded_positions[knot].tolist()])
      self.previous_racket = state.racket_pos.copy()
    self.history.append([float(self.elapsed), *np.asarray(self.position, dtype=float).tolist()])

  def sensor_due(self, now):
    """Emit only arrived observations of the physical ball's past positions."""
    while True:
      i = self.next_frame
      capture = self.capture_times[i] if i < len(self.capture_times) else self.capture_times[-1]+(i-len(self.capture_times)+1)/180.
      if i < len(self.capture_times):
        original_stamp, arrival = self.sensor_times[i]
      else:
        # Once the incoming recording ends, retain a100Hz batched sensor with
        # fresh simulated observations, never replay real outgoing positions.
        original_stamp = math.ceil((capture+self.reconstructed_clock_shift)/.01)*.01
        original_stamp += (i % 2)*.00003
        original_stamp = max(original_stamp, self.last_sensor_stamp+.00003)
        arrival = original_stamp+.001
      if now+1e-10 < arrival or now+1e-10 < capture:
        return
      history = np.asarray(self.history)
      position = np.array([np.interp(capture, history[:, 0], history[:, j]) for j in range(1, 4)])
      stamp = original_stamp if self.mode == "recorded" else capture
      yield capture, float(stamp), position+self.translation
      self.last_sensor_stamp = float(original_stamp)
      self.next_frame += 1

  def return_report(self):
    return {**super().return_report(), "recorded_case": self.case["id"],
            "handoff_reason": self.release_reason,
            "spawn_time": self.spawn_time,
            "physical_history": self.history}


class RecordedFeeder:
  independent_launch = True
  requires_motion_geometry = False

  def __init__(self, cfg, options, _rng):
    self.cfg, self.options = cfg, options
    self.cases = json.loads(Path(options["recording"]).read_text())["cases"]
    self.index = 0

  def sample(self, _unused=None):
    case = self.cases[self.index % len(self.cases)]
    self.index += 1
    return RecordedBall.create(self.cfg, case, np.asarray(self.options["world_translation"]),
                               self.options["sensor_clock"])


class PositionOnlyMocap(SimulatedMocap):
  def publish(self, kind, position, quaternion=None):
    # Suppress default100Hz ball publication. Custom capture/delivery happens
    # below, retaining the original header/arrival timing independently.
    if kind != "ball":
      super().publish(kind, position, quaternion)


class RecordedRuntime(MujocoRuntime):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.metadata.update(ball_plant="recorded incoming positions; physical collision handoff",
                         privileged_planner_inputs=False, robot_per_ball_reset=False)
    self.video = self.renderer = None
    self.next_video = 0.
    video_path = self.options["feed"].get("video")
    if video_path:
      import imageio.v2 as imageio
      self.video = imageio.get_writer(video_path, fps=25, codec="libx264", quality=7)
      self.renderer = mujoco.Renderer(self.robot.model, height=640, width=960)
      self.camera = mujoco.MjvCamera()
      self.camera.lookat[:] = [1.2, 0., .9]
      self.camera.distance, self.camera.azimuth, self.camera.elevation = 5.7, 125., -20.

  def open_mocap(self, config):
    self.mocap = PositionOnlyMocap(config, self)
    self.publish_root()
    return self.mocap

  def sample_lowstate(self):
    super().sample_lowstate()
    if self.ball is None:
      return
    if self.ball.spawn_time is None:
      self.ball.spawn_time = self.monotonic()-self.ball.elapsed
    for _capture, stamp, position in self.ball.sensor_due(self.ball.elapsed):
      self.mocap.accept(dict(kind="ball", frame="hope_table",
        stamp=10000+self.ball.spawn_time+stamp, position=position.tolist()),
        wall_time=self.time(), monotonic_time=self.monotonic())

  def record(self, row):
    if self.served >= self.options["feed"]["count"]:
      self.continuous_feed, self.next_serve, self.force_ball = False, None, False
    super().record(row)
    if not self.renderer or self.ball is None or self.monotonic() < self.next_video:
      return
    from PIL import Image, ImageDraw
    self.renderer.update_scene(self.robot.data, self.camera)
    for ball in self.balls:
      scene = self.renderer.scene
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                          np.full(3, self.cfg["ball"]["ball_radius"]), ball.position,
                          np.eye(3).ravel(), np.array([1., .6, .1, 1.]))
      scene.ngeom += 1
    image = Image.fromarray(self.renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 46), fill=(15, 15, 15))
    draw.text((12, 8), f"Recorded incoming ball -> estimator -> A3 -> motion selector -> G1 policy | {self.ball.mode}", fill="white")
    draw.text((12, 27), f"Serve {self.served}: {self.ball.case['id']} | {row['state']} | t={self.monotonic():.2f}s", fill="white")
    self.video.append_data(np.asarray(image))
    self.next_video = self.monotonic()+.04-1e-8

  def close(self):
    super().close()
    if self.renderer:
      self.renderer.close()
      self.renderer = None
    if self.video:
      self.video.close()
      self.video = None


def analyze(output):
  summary = json.loads((output/"summary.json").read_text())
  rows = [json.loads(line) for line in (output/"trace.jsonl").open()]
  from types import SimpleNamespace
  from deploy.hope_g1.world import mujoco_ball_config
  translation = np.asarray(summary["simulation_world_translation"])
  config = mujoco_ball_config(summary["rally_runtime_config"]["ball"], translation)
  source = json.loads(Path(summary["rally_runtime_config"]["feed"]["recording"]).read_text())
  selections = [event for event in summary["events"] if event["event"] == "motion_selected"]
  hit_planes = {float(event["raw_command"].get("hit_plane_x", event["raw_command"]["position"][0]))
                for event in selections}
  crossings = {}
  remote_racket = SimpleNamespace(racket_pos=np.array([-100., 0., 0.]), racket_normal=np.array([1., 0., 0.]))
  for case in source["cases"]:
    for hit_plane in hit_planes:
      probe = RecordedBall.create(config, case, translation, "recorded")
      for _ in range(1600):
        previous = probe.position.copy()
        probe.substep(remote_racket, .001)
        plane = hit_plane-translation[0]
        if previous[0] > plane >= probe.position[0]:
          fraction = (previous[0]-plane)/(previous[0]-probe.position[0])
          crossings[case["id"], hit_plane] = [probe.elapsed-.001+fraction*.001,
            *(previous+fraction*(probe.position-previous)+translation).tolist()]
          break
  predictions = []
  for event in selections:
    command = event["raw_command"]
    current = min(rows, key=lambda row: abs(row["time"]-event["deadline"]))
    # This uses independent physical ball history, never the planner's fit.
    ball = max((b for b in summary["ball_returns"] if b["spawn_time"] <= event["time"]),
               key=lambda b: b["spawn_time"])
    hit_plane = float(command.get("hit_plane_x", command["position"][0]))
    truth = crossings.get((ball["recorded_case"], hit_plane))
    history = np.asarray(ball["physical_history"])
    elapsed = event["deadline"]-ball["spawn_time"]
    physical_ball = np.array([np.interp(elapsed, history[:, 0], history[:, j]) for j in range(1, 4)])+translation
    predictions.append(dict(case=ball["recorded_case"], time=event["time"], motion=event["motion_index"],
      deadline=event["deadline"], goal_position=command["position"],
      tts=command.get("remaining", event["deadline"]-event["time"]),
      predicted_bounces=command.get("predicted_bounces", command.get("num_bounces")),
      hit_plane_x=hit_plane,
      ball_position_at_deadline=physical_ball.tolist(),
      unperturbed_crossing=truth,
      position_prediction_error_m=(float(np.linalg.norm(np.asarray(command["position"])-truth[1:]))
                                   if truth is not None else None),
      time_prediction_error_s=(float(event["deadline"]-ball["spawn_time"]-truth[0])
                               if truth is not None else None),
      actual_contact=ball["contact"], legal_return=ball["legal_return"],
      prediction_truth_note="Separate unperturbed plant rollout; diagnostic only, never provided to planner",
      deadline_racket_position=current.get("racket_pos"),
      racket_goal_error_m=float(np.linalg.norm(np.asarray(current["racket_pos"])-command["position"]))))
  return dict(run=str(output), hits=summary["simulated_hits"], balls=summary["simulated_balls"],
    legal_returns=summary["simulated_legal_returns"], completed=summary["completed_swings"],
    recovered=summary["recovered_swings"], failure=summary["failure"], error=summary["error"],
    predictions=predictions)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--recording", type=Path, default=DEFAULT_INPUT)
  p.add_argument("--sensor-clock", choices=["recorded", "capture"], default="recorded")
  p.add_argument("--render", action="store_true")
  p.add_argument("--manual", action="store_true")
  p.add_argument("--video", action="store_true")
  p.add_argument("--duration", type=float, default=45.)
  p.add_argument("--repetitions", type=int, default=1)
  p.add_argument("--output", type=Path, required=True)
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(p)
  args = p.parse_args()
  configure_viewer(args)
  if args.manual and not args.render:
    p.error("manual controls require render")
  if args.repetitions < 1:
    p.error("repetitions must be positive")
  if (args.output/"summary.json").exists():
    p.error("Use a fresh output directory")
  args.output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  rally = yaml.safe_load(Path("deploy/hope_g1/configs/rally_mujoco_random.yaml").read_text())
  source = json.loads(args.recording.read_text())
  count = len(source["cases"])*args.repetitions
  rally["feed"] = dict(mode="recorded", recording=str(args.recording.resolve()),
    world_translation=rally["world"]["mujoco_to_world_translation"],
    sensor_clock=args.sensor_clock, continuous_on_policy_entry=True,
    interval_seconds=5., count=count,
    video=str((args.output/"humanoid.mp4").resolve()) if args.video else None)
  resolved = args.output/"rally_config.yaml"
  resolved.write_text(yaml.safe_dump(rally, sort_keys=False))
  config = yaml.safe_load(Path("deploy/hope_g1/configs/task.yaml").read_text())
  config.pop("task")
  deployment = load_deployment(**config, ball_config=rally["ball"], world_config=rally["world"])
  with patch("deploy.g1_pingpong.simulation_runtime.MujocoRuntime", RecordedRuntime):
    result = run_deployment(deployment, "deploy/hope_g1/configs/real.yaml",
      "deploy/hope_g1/configs/mocap.yaml", simulation_config=resolved,
      controller_factory=HopeController, motion_index=0, motion_select="auto",
      motion_allow=list(range(len(deployment.motions))), duration=args.duration,
      episodes=count, render=args.render, controls="manual" if args.manual else "auto",
      output=args.output, seed=7, feeder_factory=RecordedFeeder, policy_wait=1.)
  diagnostic = analyze(args.output)
  (args.output/"diagnosis.json").write_text(json.dumps(diagnostic, indent=2))
  print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
  main()

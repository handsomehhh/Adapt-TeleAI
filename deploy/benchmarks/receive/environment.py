"""Stage-two rollout: position observations only enter the receiving agent."""

import json
import time
from pathlib import Path

import numpy as np

from deploy.benchmarks.racket_command.environment import MujocoCommandEnvironment
from deploy.benchmarks.racket_command.metrics import evaluate
from deploy.benchmarks.racket_command.physical_ball import PhysicalBall
from deploy.benchmarks.racket_command.sampling import Trial
from deploy.benchmarks.racket_command.visualization import draw_scene
from deploy.g1_pingpong.perception import BallEstimator


class ReceiveEnvironment(MujocoCommandEnvironment):
  def __init__(self, agent, config, output, estimator_config, **kwargs):
    super().__init__(agent, config, output, **kwargs)
    self.estimator_config = estimator_config
    self.current_feed = None
    self.current_trial = None

  def display(self, trial, samples, decision=None):
    if self.viewer is None:
      return
    if not self.viewer.is_running():
      self.quit = True
      return
    self.current_trial = trial
    with self.viewer.lock():
      draw_scene(
        self.viewer.user_scn,
        self.config,
        trial,
        samples,
        self.agent.translation,
        self.completed,
        ball=self.ball,
      )
    t = samples[-1]["time"] if samples else 0
    command = self.agent.latched_command
    target = (
      f"p={np.round(command['position'], 3).tolist()}, v={np.round(command['velocity'], 2).tolist()}"
      if command
      else "Waiting for an accepted planner command"
    )
    text = f"Central receiving test\n{self.current_feed['trial_id']} / {self.current_feed['cell_y']},{self.current_feed['cell_z']}\n{t:.3f}s\n{target}\n"
    text += f"{command['deadline'] - t:+.3f}s" if command else "Pending"
    text += f"\ncontact={self.ball.contact is not None}, net_hit={self.ball.net_hit}, phase={samples[-1]['phase']}"
    self.viewer.set_texts(
      (
        None,
        None,
        "Stage 2: independent random ball\nTrial / incoming cell\nTime\nBall estimate -> A3 planner\nRemaining to strike\nPhysical ball\nGold=planner target, cyan=actual racket\nOrange=physical ball; Space=pause, Q=quit\nPlayback [ / ] / \\",
        text + f"\n{self.playback.rate:g}x",
      )
    )
    self.viewer.sync()

  def run_ball(self, feed, ordinal):
    self.current_feed = feed
    self.current_trial = None
    self.agent.reset()
    self.ball = PhysicalBall(
      feed["initial_position"], feed["initial_velocity"], self.config["physical_ball"]
    )
    self.ball.feed = dict(source="independent_random_launch", **feed)
    samples = [self.agent.measure()]
    self.playback.reset(samples[0]["time"])
    self.ball.previous_racket = np.array(samples[0]["position"])
    estimator = BallEstimator(**self.estimator_config)
    rng = np.random.default_rng(
      np.random.SeedSequence([self.config["seed"], feed["trial_id"], 782])
    )
    observations = []
    next_observation = 0.0
    interval = 1 / self.config["observation_hz"]
    failure = None
    captures = set()

    def observe():
      nonlocal next_observation
      if self.ball.contact is None and self.ball.elapsed + 1e-10 >= next_observation:
        p = self.ball.position + rng.normal(0, self.config["position_noise_std"], 3)
        estimator.update(p, self.ball.elapsed, received=self.ball.elapsed)
        observe_ball = getattr(self.agent, "observe_ball", None)
        if observe_ball is not None:
          observe_ball(p, self.ball.elapsed, self.ball.elapsed)
        observations.append(dict(time=self.ball.elapsed, position=p.tolist()))
        next_observation += interval

    def physics_step(dt, p, n):
      self.ball.step(dt, p, n)
      observe()

    observe()
    while samples[-1]["time"] < self.config["episode_seconds"]:
      while self.paused and not self.quit:
        self.display(self.current_trial, samples)
        time.sleep(0.02)
      if self.quit:
        failure = "interrupted"
        break
      wall = time.monotonic()
      estimate = estimator.get(samples[-1]["time"])
      try:
        self.agent.advance_ball(estimate, on_physics_step=physics_step)
        sample = self.agent.measure()
        samples.append(sample)
        if self.agent.lost_standing(sample):
          failure = "lost_standing"
          break
      except (ValueError, RuntimeError, TimeoutError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        break
      command = self.agent.latched_command
      if command and self.current_trial is None:
        self.current_trial = Trial(
          feed["trial_id"],
          feed["cell_y"],
          feed["cell_z"],
          tuple(command["position"]),
          tuple(command["normal"]),
          tuple(command["velocity"]),
          command["deadline"],
        )
      self.display(self.current_trial, samples, self.agent.decision)
      if self.renderer and ordinal < self.capture_first:
        events = {
          "launch": len(samples) == 2,
          "command": command is not None,
          "deadline": command is not None and sample["time"] >= command["deadline"],
          "contact": self.ball.contact is not None,
          "landing": self.ball.first_landing is not None,
        }
        for tag, ready in events.items():
          if ready and tag not in captures:
            self.capture(
              self.current_trial, samples, f"ball_{feed['trial_id']:04d}_{tag}"
            )
            captures.add(tag)
      if self.viewer:
        self.playback.wait(sample["time"], frame_started=wall)
    command = self.agent.latched_command
    tracking = (
      evaluate(
        self.current_trial,
        samples,
        True,
        bool(samples[-1].get("recovered")),
        failure,
        self.config["tolerances"],
      )
      if command
      else dict(
        accepted=False,
        target_pass=False,
        success=False,
        deadline_observed=False,
        failure=failure,
      )
    )
    physical = self.ball.report()
    row = dict(
      trial_id=feed["trial_id"],
      cell_y=feed["cell_y"],
      cell_z=feed["cell_z"],
      feed=feed,
      command=command,
      tracking=tracking,
      accepted=command is not None,
      motion_index=command["motion_index"] if command else None,
      recovered=bool(samples[-1].get("recovered")),
      failure=failure,
      ball=physical,
      ball_hit=physical["hit"],
      legal_return=physical["legal_return"],
      success=bool(
        physical["legal_return"] and failure is None and samples[-1].get("recovered")
      ),
      max_tilt_degrees=max(s["tilt_degrees"] for s in samples),
    )
    if command:
      row["planner_position_error_m"] = float(
        np.linalg.norm(np.array(command["position"]) - feed["oracle_crossing_position"])
      )
      row["planner_time_error_s"] = command["deadline"] - feed["oracle_crossing_time"]
      diagnostic = command["diagnostic"]
      row["planned_net_clear"] = bool(
        diagnostic.get("planned_net_clear", diagnostic.get("clears_net", False))
      )
      row["planned_bypasses_posts"] = bool(
        diagnostic.get("bypasses_posts", diagnostic.get("bypasses_net_posts", False))
      )
    self.completed.append(row)
    folder = Path(self.output) / "traces"
    folder.mkdir(exist_ok=True)
    (folder / f"ball_{feed['trial_id']:04d}.json").write_text(
      json.dumps(
        dict(
          samples=samples,
          ball_observations=observations,
          planner_records=self.agent.planner_records,
        )
      )
    )
    if self.viewer:
      until = time.monotonic() + self.config["display_hold_seconds"]
      while time.monotonic() < until and not self.quit:
        self.display(self.current_trial, samples)
        time.sleep(0.02)
    return row

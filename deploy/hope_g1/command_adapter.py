"""HOPE implementation of the benchmark agent interface; no target sampling.

Bootstrap through the existing suspended/AMP/policy entry once, then restore
the complete prepared MuJoCo and controller state before every independent trial.
"""

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import yaml

from deploy.g1_pingpong.core import rotation
from deploy.g1_pingpong.simulation_runtime import MujocoRuntime, _run_attempt
from deploy.hope_g1.configuration import load_deployment
from deploy.hope_g1.controller import HopeController


class HopeCommandAdapter:
  def __init__(
    self,
    task_config,
    rally_config,
    runtime_config,
    mocap_config,
    output,
    home_index=0,
    allow=None,
  ):
    task = yaml.safe_load(Path(task_config).read_text())
    task.pop("task")
    self.options = yaml.safe_load(Path(rally_config).read_text())
    self.options["target_source"] = "a3_planner"
    self.options["a3"]["target_mode"] = "raw"
    self.runtime = yaml.safe_load(Path(runtime_config).read_text())
    # Avoid unrelated high-rate IO journals during simulation-only preparation.
    self.runtime.update(
      record_lowstate_hz=0, record_command_writes=False, latency_auto_summary=False
    )
    self.deployment = load_deployment(
      **task, ball_config=self.options["ball"], world_config=self.options["world"]
    )
    cfg = copy.deepcopy(self.deployment.settings)
    cfg["physics_dt"] = 0.001
    cfg["decimation"] = round(cfg["control_dt"] / 0.001)
    cfg["sim_options"]["timestep"] = 0.001
    self.deployment = replace(self.deployment, settings=cfg)
    self.dt = cfg["control_dt"]
    self.translation = np.asarray(cfg["simulation_world_translation"])
    self.session = MujocoRuntime(
      self.deployment,
      self.runtime,
      self.options,
      controls="auto",
      render=False,
      seed=0,
      episodes=1,
      standing_only=True,
      stable_seconds=1,
      policy_wait=0,
      final_hold=0,
      startup="suspended",
      ball_source="sim",
    )
    self.robot = self.session.robot
    self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
    original_record = self.session.record
    ready_since = None
    self.baseline = None

    def record(row):
      nonlocal ready_since
      original_record(row)
      if row["state"] != "ARMED":
        return
      now = self.session.monotonic()
      ready_since = now if ready_since is None else ready_since
      if self.baseline is None and now - ready_since >= 1.0:
        if self.session.physics_validation["first_loss_of_standing_time"] is not None:
          raise RuntimeError("Preparation lost standing")
        self.snapshot = np.empty(mujoco.mj_stateSize(self.robot.model, self.spec))
        mujoco.mj_getState(self.robot.model, self.robot.data, self.snapshot, self.spec)
        self.start_time = now
        self.baseline = self.clone(self.session.engine)
        self.baseline.receiving = True
        self.session.finished_since = now

    self.session.record = record

    class CommandOnlyController(HopeController):
      def step(self, state, estimate, now, **kwargs):
        # The shared entry flow sets receiving=True at UP. Ignore estimates
        # explicitly during preparation; standing_only also disables the feeder.
        return super().step(state, None, now, **kwargs)

    def factory(*args, **kwargs):
      return CommandOnlyController(*args, **kwargs)

    preparation = _run_attempt(
      self.deployment,
      self.runtime,
      mocap_config,
      session=self.session,
      simulation_config=rally_config,
      motion_index=home_index,
      motion_select="auto",
      motion_allow=allow,
      duration=40,
      output=Path(output) / "preparation",
      controller_factory=factory,
    )
    if self.baseline is None or preparation.get("failure"):
      raise RuntimeError(f"Could not prepare baseline: {preparation.get('failure')}")
    self.metadata = dict(
      task="HOPEPingPongMultiMotionG1",
      checkpoint=str(self.deployment.checkpoint),
      selector="HopeController.select_racket_command (unchanged A3 position scoring)",
      policy="HopeActorController, saved 175D->29D checkpoint",
      target_mode="raw",
      preparation="Existing START/AMP/UP flow, 1s ARMED hold",
      reset="Full MuJoCo integration state and prepared actor/controller history",
      control_dt=self.dt,
      physics_dt=0.001,
      world_frame="hope_table",
      mujoco_to_world_translation=self.translation.tolist(),
      selector_max_tts=self.options["a3"]["max_command_time_to_strike"],
      selector_max_start_frame=self.options["max_start_frame"],
      selector_max_intercept_distance=self.options["max_intercept_distance"],
      joint_kp=self.robot.cfg["robot"]["kp"],
      joint_kd=self.robot.cfg["robot"]["kd"],
      actual_network_connections=False,
      actual_robot_commands_sent=False,
      observation_source="Ideal MuJoCo kinematics and velocities; no sensor latency/noise",
    )
    self.reset()

  @staticmethod
  def clone(engine):
    # The inference network is stateless; controller history is cloned separately.
    actor = engine.controller.policy
    shared = [actor, engine.library, engine.home, *(engine.references or [])]
    if engine.standing.home is not None:
      shared.append(engine.standing.home)
    memo = {id(v): v for v in shared}
    if engine.command_planner is not None:
      from deploy.benchmarks.planner_snapshot import PlannerSnapshot

      memo[id(engine.command_planner)] = PlannerSnapshot(engine.command_planner)
    if engine.physical_limit_observer is not None:
      memo[id(engine.physical_limit_observer)] = None
    return copy.deepcopy(engine, memo)

  def reset(self):
    self.engine = self.clone(self.baseline)
    self.robot.set_gains(self.robot.cfg["robot"]["kp"], self.robot.cfg["robot"]["kd"])
    mujoco.mj_resetData(self.robot.model, self.robot.data)
    mujoco.mj_setState(self.robot.model, self.robot.data, self.snapshot, self.spec)
    mujoco.mj_forward(self.robot.model, self.robot.data)
    self.ticks = 0
    self.accepted = False
    self.decision = None

  def state(self):
    state = self.robot.state()
    for key in ("root_pos", "anchor_pos", "racket_pos"):
      setattr(state, key, getattr(state, key) + self.translation)
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(
      self.robot.model,
      self.robot.data,
      mujoco.mjtObj.mjOBJ_BODY,
      self.robot.root,
      velocity,
      0,
    )
    state.root_velocity_world = velocity[3:].copy()
    return state

  def advance(self, command=None, on_physics_step=None):
    now = self.start_time + self.ticks * self.dt
    if command is not None:
      # Convert trial-relative deadline to the prepared controller clock.
      command = copy.deepcopy(command)
      command.deadline += self.start_time
    target = self.engine.step(self.state(), None, now, racket_command=command)
    if command is not None:
      self.decision = copy.deepcopy(self.engine.debug["decision"])
      self.accepted = self.engine.lifecycle.state == "EXECUTE"
    self.integrate(target, on_physics_step)

  def integrate(self, target, on_physics_step=None):
    """Shared unmodified PD/physics path for both benchmark stages."""
    self.robot.set_target(target)
    for _ in range(self.robot.cfg["decimation"]):
      self.robot.substep()
      if on_physics_step is not None:
        on_physics_step(
          self.robot.cfg["physics_dt"],
          self.robot.data.site_xpos[self.robot.racket].copy() + self.translation,
          self.robot.data.site_xmat[self.robot.racket].reshape(3, 3)[:, 1].copy(),
        )
    self.ticks += 1

  def measure(self):
    state = self.state()
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(
      self.robot.model,
      self.robot.data,
      mujoco.mjtObj.mjOBJ_SITE,
      self.robot.racket,
      velocity,
      0,
    )
    tilt = np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
    return dict(
      time=self.ticks * self.dt,
      position=state.racket_pos.tolist(),
      normal=state.racket_normal.tolist(),
      velocity=velocity[3:].tolist(),
      tilt_degrees=float(tilt),
      root_height=float(state.root_pos[2]),
      phase=self.engine.lifecycle.state,
      recovered=self.engine.recovered > 0,
      motion_index=self.engine.motion_index,
    )

  def lost_standing(self, sample):
    physical_height = sample["root_height"] - self.translation[2]
    return sample["tilt_degrees"] > 45 or physical_height < 0.4

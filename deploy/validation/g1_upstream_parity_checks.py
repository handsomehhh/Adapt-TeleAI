"""Compare deployment with G1 planning with only lateral rejection disabled.

Requires the sibling PingPong and g1_planner_ros2 repositories (or set PINGPONG_REPO
and G1_PLANNER_PACKAGE). Run with the Adapt-TeleAI Python environment.
"""

# ruff: noqa: E402, I001
# Importing play_1v0 establishes its repository-owned reference imports.
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINING = Path(os.environ.get("PINGPONG_REPO", ROOT.parent / "PingPong"))
PLANNER = Path(
  os.environ.get(
    "G1_PLANNER_PACKAGE", ROOT.parent / "g1_planner_ros2/src/hope_planner_g1"
  )
)
sys.path.insert(0, str(TRAINING / "hope_training/mjlab_training/src"))
sys.path.insert(0, str(PLANNER))

from hope_mjlab.play_1v0 import CheckpointPolicy, build_observation
from hope_mjlab.play_multiplayer import MultiPlanner
from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.racket_target_planner import RacketTargetPlanner
from a3_deploy_onnx_ref_pingpong.ball_observation import BallObservationController
from hope_planner_g1.planner import G1Planner
from hope_planner_g1.observation import BallObserver
from deploy.hope_g1_hitter.configuration import load_deployment
from deploy.hope_g1_hitter.controller import HitterActor, observation

ARTIFACT = ROOT / "deploy/hope_g1_hitter/artifacts/hitter_extended_20261103_10700"
CONTRACT = json.loads((ARTIFACT / "contract.json").read_text())


def upstream(physics):
  # Call the original plan()/outgoing_velocity()/target_station() methods;
  # only replace physical calibration and supply fixed 1v1 defaults.
  p = MultiPlanner.__new__(MultiPlanner)
  p.physics, p.table = physics, TableParams()
  p.config = PlannerConfig(max_predict_time=2.6, off_table_floor_z=-p.table.height)
  p.predictor = BallTrajectoryPredictor(p.physics, p.config, p.table)
  p.planner = RacketTargetPlanner(p.physics, p.config, p.table)
  p.controller = SimpleNamespace(
    boxes=np.asarray(CONTRACT["racket_pos_range_per_clip"]),
    velocity_boxes=np.asarray(CONTRACT["racket_vel_range_per_clip"]),
    cfg=SimpleNamespace(
      unrestricted_planner=False, own_half_bounces_only=False, command_profile=None,
      planner_limits=SimpleNamespace(enabled=lambda name: name != "lateral"),
    ),
    station=np.array([-0.65, -0.45]),
    side=0,
  )
  p.args = SimpleNamespace(
    return_flight_time=0.5, landing_x=2.055, aim_at_opponent=False
  )
  p.next_controller = p.opponent_controller = None
  p.landing_target = SimpleNamespace(resolve_local=lambda xy: xy)
  return p


@pytest.mark.parametrize("calibrated", [False, True])
def test_multiseed_planner_matches_only_lateral_relaxed(calibrated):
  physics = (
    BallPhysics(k=0.096, C_h=0.816019, C_v=0.862769) if calibrated else BallPhysics()
  )
  old, new = upstream(physics), G1Planner(CONTRACT, physics=physics)
  counts = [0, 0, 0]
  for seed in range(8):
    rng = np.random.default_rng(seed)
    for i in range(40):
      bounces = i % 2
      position = (
        rng.uniform([-0.02, -1.4, 0.03], [0.65, -0.15, 0.5])
        if bounces
        else rng.uniform([1.5, -1.4, 0.1], [2.5, -0.1, 0.7])
      )
      velocity = rng.uniform([-6, -0.8, -1.5], [-2.5, 0.8, 1.5])
      station = np.array([-0.65, rng.uniform(-0.9, -0.35)])
      side = int(rng.integers(2))
      ref = old.plan(position + [0, 0, 0.76], velocity, station, side, bounces=bounces)
      actual = new.plan(position, velocity, station, side, stamp=123.0, bounces=bounces)
      assert (ref is None) == (actual is None), (
        seed,
        i,
        old.last_plan_diagnostic,
        new.diagnostic,
      )
      if ref is None:
        counts[2] += 1
        continue
      counts[actual.side] += 1
      assert ref.side == actual.side
      np.testing.assert_allclose(actual.position + [0, 0, 0.76], ref.target, atol=1e-9)
      np.testing.assert_allclose(actual.velocity, ref.racket_velocity, atol=1e-5)
      np.testing.assert_allclose(actual.station, ref.station, atol=1e-9)
      assert actual.strike_time - 123.0 == pytest.approx(ref.flight_time, abs=1e-9)
  assert counts[0] > 10 and counts[1] > 10 and counts[2] > 10
  print(f"calibrated={calibrated}: 320 cases, FH/BH/rejected={counts}")


def test_position_estimator_and_bounce_tracking_match_original():
  physics = BallPhysics(k=0.096, C_h=0.816019, C_v=0.862769)
  old = upstream(physics)
  reference = BallObservationController([old], [old.controller], [0])
  actual = BallObserver(G1Planner(CONTRACT, physics=physics))
  # Piecewise flight with one bounce, then a longitudinal reversal/new flight.
  for i in range(300):
    t = i / 300
    x = 1.2 - 3 * t if t < 0.7 else -0.9 + 3 * (t - 0.7)
    z = 0.02 + 0.9 * abs(t - 0.3) - 0.3 * (t - 0.3) ** 2
    p = np.array([x, -0.8, z])
    reference.observe(10 + t, p + [0, 0, 0.76])
    actual.observe(10 + t, p)
    assert actual.bounces == reference.bounces
    assert actual.direction == reference.direction
    assert actual.estimator.ready == reference.estimator.ready
    if actual.estimator.ready:
      for a, b in zip(
        actual.estimator.estimate(), reference.estimator.estimate(), strict=True
      ):
        np.testing.assert_allclose(a, b, atol=1e-10)
  assert actual.direction == 1
  actual.observe(12.0, p)
  reference.observe(12.0, p + [0, 0, 0.76])
  assert actual.bounces == reference.bounces == 0
  assert not actual.estimator.ready and not reference.estimator.ready


def test_observation_actor_and_action_parity():
  torch.set_num_threads(1)
  config_dir = ROOT / "deploy/hope_g1_hitter/configs"
  opts = yaml.safe_load((config_dir / "rally.yaml").read_text())
  deployment = load_deployment(
    ARTIFACT, ball_config=opts["ball"], world_config=opts["world"]
  )
  ref_policy = CheckpointPolicy(ARTIFACT / "model_10700.pt")
  actor = HitterActor(deployment.settings, deployment.actor)
  rng = np.random.default_rng(10700)
  default = np.asarray(CONTRACT["default_q"])
  for _ in range(100):
    q = default + rng.normal(0, 0.1, 29)
    qd, gyro, last = (
      rng.normal(0, 0.5, 29),
      rng.normal(0, 0.2, 3),
      rng.normal(0, 0.3, 29).astype(np.float32),
    )
    root = np.array([-0.65, -0.45, 0.1]) + rng.normal(0, 0.01, 3)
    quat = np.r_[1.0, rng.normal(0, 0.1, 3)]
    quat /= np.linalg.norm(quat)
    station = np.array([-0.65, -0.5])
    target, velocity, tts = (
      np.array([-0.2, -0.8, 0.15]),
      np.array([1.5, 0.1, 0.8]),
      float(rng.uniform(-1, 0.8)),
    )
    state = SimpleNamespace(q=q, dq=qd, gyro=gyro, root_pos=root, root_quat=quat)
    # The reference uses floor world; both root and contact gain +0.76 m.
    ref_state = SimpleNamespace(
      q=q, qd=qd, base_ang_vel_b=gyro, base_pos_w=root + [0, 0, 0.76], base_quat_w=quat
    )
    ref_target = SimpleNamespace(
      pos_w=target + [0, 0, 0.76], vel_w=velocity, time_to_strike=tts
    )
    expected = build_observation(ref_state, ref_target, last, default, station)
    actual = observation(state, last, default, station, target, velocity, tts)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    action = ref_policy.infer(expected)
    np.testing.assert_allclose(deployment.actor(actual), action, atol=1e-5)
    actor.previous_joint = last.copy()
    decoded = actor.step(state, station, target, velocity, tts)
    np.testing.assert_allclose(
      decoded,
      np.clip(
        default + np.asarray(CONTRACT["action_scale"]) * action,
        actor.lower,
        actor.upper,
      ),
      atol=1e-5,
    )
    np.testing.assert_allclose(actor.previous_joint, action, atol=1e-5)


def test_deployed_robot_fk_inertia_and_collision_match_training():
  import mujoco
  from hope_mjlab import g1_robot

  options = yaml.safe_load(
    (ROOT / "deploy/hope_g1_hitter/configs/rally.yaml").read_text()
  )
  actual = load_deployment(
    ARTIFACT, ball_config=options["ball"], world_config=options["world"]
  ).model
  reference = g1_robot.robot_spec().compile()
  for body in (reference.body(i) for i in range(reference.nbody)):
    if body.name == "world":
      continue
    deployed = actual.body("robot/" + body.name)
    for field in ("mass", "inertia", "ipos", "iquat"):
      np.testing.assert_allclose(
        getattr(deployed, field), getattr(body, field), atol=1e-10
      )
  ref_geoms = [
    reference.geom(i) for i in range(reference.ngeom) if reference.geom(i).group == 3
  ]
  deployed_geoms = [
    actual.geom(i)
    for i in range(actual.ngeom)
    if actual.geom(i).group == 3
    and actual.body(int(actual.geom(i).bodyid[0])).name.startswith("robot/")
  ]
  assert len(ref_geoms) == len(deployed_geoms)
  for a, b in zip(ref_geoms, deployed_geoms, strict=True):
    for field in ("pos", "quat", "size", "contype", "conaffinity", "friction"):
      np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-10)
  rng = np.random.default_rng(242)
  ref_data, data = mujoco.MjData(reference), mujoco.MjData(actual)
  for _ in range(20):
    q = np.asarray(CONTRACT["default_q"]) + rng.normal(0, 0.1, 29)
    root = np.array([-0.65, -0.45, 0.85])
    quat = np.r_[1.0, rng.normal(0, 0.1, 3)]
    quat /= np.linalg.norm(quat)
    for model, d, prefix in ((reference, ref_data, ""), (actual, data, "robot/")):
      for name, value in zip(CONTRACT["joint_names"], q, strict=True):
        d.qpos[model.joint(prefix + name).qposadr] = value
      root_joint = model.body(prefix + "pelvis").jntadr[0]
      addr = model.jnt_qposadr[root_joint]
      d.qpos[addr : addr + 7] = np.r_[root, quat]
      mujoco.mj_forward(model, d)
    for field in ("xpos", "xmat"):
      np.testing.assert_allclose(
        getattr(data.site("robot/right_racket_surface"), field),
        getattr(ref_data.site("right_racket_surface"), field),
        atol=1e-10,
      )

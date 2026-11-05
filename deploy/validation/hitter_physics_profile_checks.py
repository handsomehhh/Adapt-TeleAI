"""Backend-selected physics, production calibration and ROS mode transitions."""

import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
from deploy.hope_g1_hitter.configuration import resolve_physics
from hope_planner_g1 import node
from hope_planner_g1.planner import G1Planner
from hope_planner_g1.profiles import load_profile, real_sources

PACKAGE = Path(node.__file__).resolve().parents[1]
CONTRACT = json.loads((PACKAGE / "config/policy_contract.json").read_text())


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_backend_profile_controls_plant_and_planner(mode):
  options = yaml.safe_load(
    Path("deploy/hope_g1_hitter/configs/rally_mujoco.yaml").read_text()
  )
  options["ball"].update(drag=999, table_restitution=999, racket_restitution=999)
  resolved = resolve_physics(options, mode)
  planner = G1Planner(CONTRACT, mode=mode)
  p, c, ball = planner.physics, planner.config, resolved["ball"]
  assert resolved["runtime_mode"] == mode
  assert (
    resolved["physics_profile_report"]["source_sha256"] == planner.profile.source_sha256
  )
  assert ball["drag"] == p.k
  assert ball["table_restitution"] == p.C_v
  assert ball["table_tangential_damping"] == p.C_h
  assert ball["ball_mass"] == p.mass
  assert ball["ball_radius"] == p.radius
  assert ball["racket_restitution"] == c.C_r
  assert ball["racket_radius"] == c.racket_radius
  assert ball["table_far_x"] == planner.table.length
  assert ball["table_center_y"] == -planner.table.width / 2
  # Physics selection must not change the trained command semantics.
  np.testing.assert_array_equal(planner.boxes, CONTRACT["racket_pos_range_per_clip"])
  np.testing.assert_array_equal(
    planner.velocity_boxes, CONTRACT["racket_vel_range_per_clip"]
  )
  np.testing.assert_allclose(planner.landing, [2.055, -0.7625, 0])
  assert planner.flight_time == 0.5
  if mode == "sim":
    assert (p.k, p.C_h, p.C_v, p.mass) == (0.1261, 0.64, 0.9215, 0.0034)
    assert (c.e_exp_g1, c.e_exp_g2, c.paddle_a_t, c.paddle_mu, c.k_m) == (
      0.759,
      -0.0441,
      0.52,
      0.5,
      0.00444,
    )
    assert ball["racket_contact_model"] == "g1_1v1"
  else:
    assert (p.k, p.C_h, p.C_v, p.mass, c.C_r) == (
      0.096,
      0.816019,
      0.862769,
      0.0027,
      0.6,
    )
    assert c.k_m == c.paddle_a_t == c.paddle_mu == 0
    assert ball["racket_contact_model"] == "constant_restitution"


def test_real_contact_inverts_original_a30808_solver(monkeypatch):
  # Optional reference comparison only; runtime no longer imports this repo.
  old = (
    Path(__file__).resolve().parents[3]
    / "HOPE-TeleAI/hope_ws/src/hope_planner_optimized_2_A30808"
  )
  if not old.is_dir():
    pytest.skip("Original A30808 source unavailable for optional comparison")
  monkeypatch.syspath_prepend(str(old))
  from hope_planner_optimized_2.racket_target_planner import RacketTargetPlanner

  profile = load_profile("real")
  old = RacketTargetPlanner(profile.physics, profile.config, profile.table)
  rng = np.random.default_rng(42)
  for _ in range(100):
    incoming = rng.uniform([-8, -2, -4], [-1, 2, 4])
    requested = rng.uniform([1, -2, -1], [8, 2, 5])
    racket, normal = old._compute_racket_velocity(
      incoming, requested, profile.config.C_r
    )
    actual, spin = profile.contact(incoming, racket, normal, np.zeros(3))
    np.testing.assert_allclose(actual, requested, atol=1e-12)
    np.testing.assert_array_equal(spin, np.zeros(3))


def test_real_calibration_is_loaded_from_source_including_friction(tmp_path):
  path, _, _ = real_sources()
  config = tmp_path / "config/hope_planner.yaml"
  config.parent.mkdir()
  content = yaml.safe_load(path.read_text())
  params = content["hope_planner_optimized_2"]["ros__parameters"]
  params.update(drag_k=0.12, restitution_racket=0.71, bounce_model="friction")
  config.write_text(yaml.safe_dump(content))
  planner = G1Planner(CONTRACT, mode="real", real_config=config)
  assert planner.physics.k == 0.12 and planner.config.C_r == 0.71
  assert planner.profile.source_sha256 != load_profile("real").source_sha256
  velocity = np.array([-2, 1, -3.0])
  ch = max(
    params["bounce_rolling_ratio"],
    1 - params["bounce_friction_mu"] * (1 + params["restitution_v"]) * 3 / np.sqrt(5),
  )
  np.testing.assert_allclose(
    planner.predictor._apply_bounce(velocity),
    velocity * [ch, ch, -params["restitution_v"]],
  )
  with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
    load_profile("real", config.parent / "missing.yaml")


def test_unknown_mode_has_no_fallback():
  with pytest.raises(ValueError):
    load_profile("auto")


def test_ros_feedback_selects_and_resets_profiles(monkeypatch):
  """Execute actual node callbacks with a fake ROS host (no live DDS required)."""
  errors = []

  class FakeNode:
    def __init__(self, name):
      pass

    def declare_parameter(self, key, value):
      return NS(value=value)

    def create_publisher(self, *args):
      return NS(publish=lambda msg: None)

    def create_subscription(self, *args):
      pass

    def create_timer(self, *args):
      pass

    def destroy_node(self):
      pass

    def get_logger(self):
      return NS(info=lambda msg: None, error=errors.append)

  def exercise(planner_node):
    assert planner_node.planner is None
    previous = None
    for sequence, mode in enumerate(("sim", "real", "sim"), 1):
      state = NS(
        contract_sha256=planner_node.contract_hash,
        header=NS(frame_id="world"),
        station=NS(x=-0.65, y=-0.45),
        side=0,
        session_id="test",
        sequence=sequence,
        enabled=True,
        runtime_mode=mode,
        physics_sha256=load_profile(mode).source_sha256,
      )
      planner_node.feedback(state)
      assert planner_node.state is state
      assert planner_node.active_mode == mode
      assert planner_node.observer is not previous
      assert planner_node.observer.bounces == 0
      previous = planner_node.observer
      previous.bounces = 8
    state = NS(**vars(state) | {"sequence": 4, "physics_sha256": "wrong"})
    planner_node.feedback(state)
    assert planner_node.state is None and planner_node.observer.bounces == 0
    state.runtime_mode = "unknown"
    planner_node.feedback(state)
    assert planner_node.state is None

  modules = {
    "rclpy": dict(init=lambda: None, spin=exercise, try_shutdown=lambda: None),
    "rclpy.node": dict(Node=FakeNode),
    "rclpy.qos": dict(QoSProfile=lambda **kw: kw, ReliabilityPolicy=NS(BEST_EFFORT=1)),
    "ament_index_python.packages": dict(
      get_package_share_directory=lambda name: str(PACKAGE)
    ),
    "geometry_msgs.msg": dict(PoseArray=object),
    "hope_msgs.msg": dict(G1PlannerState=object, G1RacketCommand=object),
  }
  for name, attributes in modules.items():
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
  node.main()
  assert len(errors) == 2

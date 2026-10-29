"""Recorded observation references: timing, joint order and pelvis alignment."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deploy.tools.view_real_log import (
  LogScene,
  Recording,
  Timeline,
  base_target,
  base_target_text,
  observation_reference,
)
from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_JOINT_NAMES,
)


@pytest.fixture
def recording(tmp_path):
  options = yaml.safe_load(
    (Path(__file__).parents[1] / "hope_g1/configs/rally.yaml").read_text()
  )
  summary = dict(
    control_world_frame="hope_table",
    rally_runtime_config=options,
    simulation_world_translation=options["world"]["mujoco_to_world_translation"],
    # Reverse the native order to exercise contract-based joint mapping.
    robot_command_contract=dict(joint_names=list(reversed(G1_PINGPONG_JOINT_NAMES))),
    home_motion=0,
    motion_allow=[0, 1],
    policy_files=dict(
      motions=[dict(path="/remote/home.npz"), dict(path="/remote/hit.npz")]
    ),
  )
  q = np.linspace(-0.2, 0.2, 29)
  obs = np.r_[q, np.zeros(146)].tolist()
  base = dict(
    q=np.zeros(29).tolist(),
    root_pos=[0.4, -0.6, 0.1],
    root_quat=[2, 0.2, -0.1, 0.7],
    motion_index=1,
    motion="hit.npz",
    actor_obs=obs,
    reference_frame=7,
    reference_frame_count=30,
    goal_base_xy=[0.7, -0.2],
  )
  rows = [
    dict(base, time=100, state="STABLE"),  # Stale debug data must stay hidden.
    dict(base, time=100.02, state="TO_POLICY", reference_frame=0),
    dict(base, time=100.04, state="EXECUTE"),
    dict(
      base, time=100.06, state="EXECUTE", reference_frame=8,
      actor_obs=np.r_[q + 0.3, np.zeros(146)].tolist(),
    ),
    dict(base, time=100.08, state="RECOVER", reference_frame=0),
    dict(base, time=100.1, state="ARMED", actor_obs=None),
    dict(base, time=100.12, state="STABLE"),
  ]
  (tmp_path / "summary.json").write_text(json.dumps(summary))
  (tmp_path / "trace.jsonl").write_text("\n".join(map(json.dumps, rows)))
  return Recording(tmp_path)


@pytest.mark.parametrize(
  "state", ["WAIT", "STABLE", "TO_STABLE", "HALT", "POLICY_POSTURE"]
)
def test_inactive_policy_does_not_reuse_stale_observation(state):
  q, status = observation_reference(dict(state=state, actor_obs=[0] * 175))
  assert q is None and status == "policy inactive"


@pytest.mark.parametrize(
  "obs", [None, [0] * 174, [[0] * 175], [float("nan")] * 175]
)
def test_missing_or_invalid_observation_is_not_a_pose(obs):
  assert observation_reference(dict(state="EXECUTE", actor_obs=obs))[0] is None


def test_hitter_observation_is_not_a_motion_reference():
  q, status = observation_reference(dict(state="EXECUTE", actor_obs=[0] * 104))
  assert q is None and "no reference motion" in status


def test_tracker_reference_uses_its_recorded_input():
  obs = np.arange(132, dtype=float)
  q, source = observation_reference(dict(state="EXECUTE", tracker_obs=obs))
  np.testing.assert_array_equal(q, obs[:29])
  assert source == "tracker_obs"


def test_reference_metadata_uses_home_during_entry_and_recovery(recording):
  rows = recording.rows
  assert recording.reference_text(rows[1])[0] == "home.npz"
  assert recording.reference_text(rows[2])[:2] == (
    "hit.npz", "8 / 30 (index 7, zero-based)"
  )
  assert recording.reference_text(rows[4])[:2] == (
    "home.npz", "1 / 30 (index 0, zero-based)"
  )
  assert recording.reference_text(rows[5])[:2] == ("--", "--")
  row = dict(rows[2], reference_frame=100)
  assert recording.reference_text(row)[1] == "30 / 30 (index 29, zero-based)"
  row = dict(rows[2], reference_frame=None)
  assert recording.reference_text(row)[1] == "unknown (not recorded)"
  # A swing may begin while the logged display state is still TO_POLICY.
  row = dict(rows[2], state="TO_POLICY", phase="SWING")
  assert recording.reference_text(row)[0] == "hit.npz"


def test_scene_replays_exact_input_and_aligns_full_pelvis_pose(recording):
  scene = LogScene(recording)
  timeline = Timeline(recording, start=recording.times[2], paused=True)
  row = scene.update(timeline.position)
  assert scene.reference_visible
  np.testing.assert_allclose(scene.data.qpos[scene.reference_addresses], row["reference_q"])
  np.testing.assert_array_equal(scene.data.qpos[scene.addresses], row["q"])
  measured, reference = scene.data.body("robot/pelvis"), scene.data.body("reference/pelvis")
  np.testing.assert_allclose(reference.xpos, measured.xpos, atol=1e-12)
  np.testing.assert_allclose(reference.xquat, measured.xquat, atol=1e-12)
  # Different reference joints must still produce different limb poses.
  assert not np.allclose(
    scene.data.body("reference/right_wrist_yaw_link").xpos,
    scene.data.body("robot/right_wrist_yaw_link").xpos,
  )
  reference_geoms = [
    i for i in range(scene.model.ngeom)
    if scene.model.body(scene.model.geom_bodyid[i]).name.startswith("reference/")
  ]
  np.testing.assert_allclose(scene.model.geom_rgba[reference_geoms, 3], 0.3)
  assert not np.any(scene.model.geom_contype[reference_geoms])
  assert not np.any(scene.model.geom_conaffinity[reference_geoms])
  before = scene.data.qpos.copy()
  scene.update(timeline.advance(10))  # Paused reference does not advance.
  np.testing.assert_array_equal(scene.data.qpos, before)
  scene.update(recording.times[3] - 0.001)  # Causal hold, not future frame 8.
  np.testing.assert_array_equal(scene.data.qpos, before)
  timeline.step_frames(1)
  scene.update(timeline.position)
  assert not np.allclose(scene.data.qpos[scene.reference_addresses], row["reference_q"])
  timeline.step_frames(-1)
  scene.update(timeline.position)
  np.testing.assert_array_equal(scene.data.qpos, before)
  for i in (5, 0, 6):  # Missing input, seeking before entry, leaving policy.
    scene.update(recording.times[i])
    assert not scene.reference_visible
    assert scene.data.body("reference/pelvis").xpos[2] < -100


def test_base_goal_distance_and_heading_use_same_recorded_frame(recording):
  row = dict(recording.rows[2], root_quat=[2**0.5, 0, 0, 2**0.5])
  result = base_target(row)  # Facing world +Y, target is +30cm X, +40cm Y.
  np.testing.assert_allclose(result["delta"], [0.3, 0.4])
  assert result["distance"] == pytest.approx(0.5)
  np.testing.assert_allclose(result["heading_delta"], [0.4, -0.3])
  assert base_target_text(row)[2:] == (
    "50.0 cm", "X +30.0 cm; Y +40.0 cm", "forward 40.0 cm; right 30.0 cm"
  )
  # Goal changes/return-to-home use this row, not a selected racket command.
  changed = dict(row, state="RECOVER", goal_base_xy=[0.1, -1.0])
  np.testing.assert_allclose(base_target(changed)["delta"], [-0.3, -0.4])
  np.testing.assert_allclose(base_target(row)["delta"], [0.3, 0.4])


def test_base_goal_does_not_require_reference_motion_and_rejects_stale_values(recording):
  row = dict(recording.rows[2], actor_obs=[0] * 104, reference_q=None)
  assert base_target(row) is not None
  for invalid in (None, [0], [0, float("nan")], [0, 0, 0]):
    assert base_target(dict(row, goal_base_xy=invalid)) is None
  for state in ("STABLE", "TO_STABLE", "WAIT", "HALT"):
    assert base_target(dict(row, state=state)) is None
    assert base_target_text(dict(row, state=state)) == ("--",) * 5


def test_base_goal_overlays_project_to_floor_and_keep_true_distance(recording):
  import mujoco

  from deploy.viewer import overlay_batches

  scene = LogScene(recording)
  overlay = mujoco.MjvScene(scene.model, maxgeom=200)
  t = recording.times[2]
  row = scene.update(t)
  scene.overlays(overlay, t, row)
  arrows = [g for g in overlay.geoms[:overlay.ngeom]
            if g.type == mujoco.mjtGeom.mjGEOM_ARROW]
  assert len(arrows) == 2
  arrow = arrows[0]
  current = np.r_[np.asarray(row["root_pos"])[:2] - recording.offset[:2], 0.018]
  target = np.r_[np.asarray(row["goal_base_xy"]) - recording.offset[:2], 0.018]
  np.testing.assert_allclose(arrow.pos, current, atol=1e-6)
  np.testing.assert_allclose(arrow.pos + arrow.mat[:, 2] * arrow.size[2], target, atol=1e-6)
  assert arrow.size[2] == pytest.approx(0.5)
  raised = arrows[1]
  pelvis = np.asarray(row["root_pos"]) - recording.offset
  np.testing.assert_allclose(raised.pos, pelvis, atol=1e-6)
  np.testing.assert_allclose(
    raised.pos + raised.mat[:, 2] * raised.size[2],
    [target[0], target[1], pelvis[2]], atol=1e-6,
  )
  assert raised.size[2] == pytest.approx(0.5)
  # All added shapes are supported by the web viewer too.
  batches = overlay_batches(overlay)
  np.testing.assert_allclose(batches["arrow"][0][0], current, atol=1e-6)
  scene.overlays(overlay, t, dict(row, goal_base_xy=row["root_pos"][:2]))
  assert all(g.type != mujoco.mjtGeom.mjGEOM_ARROW for g in overlay.geoms[:overlay.ngeom])
  assert all(np.isfinite(g.mat).all() for g in overlay.geoms[:overlay.ngeom])
  scene.overlays(overlay, t, dict(row, goal_base_xy=None))
  assert overlay.ngeom == 0
  scene.overlays(overlay, t, dict(row, state="STABLE"))
  assert overlay.ngeom == 0

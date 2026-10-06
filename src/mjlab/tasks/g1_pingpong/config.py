"""G1 specialization of the validated A3 two-stage ping-pong task."""

import hashlib
import json
from pathlib import Path

from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
  G1_PINGPONG_ACTION_SCALE,
  G1_PINGPONG_ANCHOR_BODY,
  G1_PINGPONG_BODY_NAMES,
  G1_PINGPONG_FEET_BODIES,
  G1_PINGPONG_JOINT_NAMES,
  G1_PINGPONG_MOUNT_OFFSET,
  G1_PINGPONG_RACKET_NORMAL,
  G1_PINGPONG_RACKET_SITE,
  G1_PINGPONG_TRACKED_BODIES,
  G1_PINGPONG_WRIST_BODY,
  get_g1_pingpong_robot_cfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.a3_pingpong.actions import FrozenTrackerActionCfg
from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommandCfg
from mjlab.tasks.a3_pingpong.config import (
  a3_receiving_env_cfg,
  a3_runner_cfg,
  a3_tracking_env_cfg,
)
from mjlab.tasks.a3_pingpong.runner import PingPongOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

DATA_ROOT = Path(__file__).resolve().parents[4] / "dataset" / "g1_pingpong"
DATA_DIR = DATA_ROOT / "motions"
MOTION_MANIFEST = DATA_ROOT / "manifest.json"
TELE_GMR_DATA_ROOT = (
  Path(__file__).resolve().parents[4] / "dataset" / "g1_pingpong_tele_gmr"
)
STAGE1_TASK = "Mjlab-PingPong-Tracking-G1-Stage1-RandomDt"
TELE_GMR_STAGE1_TASK = "Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt"
STAGE2_TASK = "Mjlab-PingPong-Receive-G1-Stage2-Adaptive"
TELE_GMR_STAGE2_TASK = "Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive"
TELE_GMR_HARD_STAGE2_TASK = "Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive"
TELE_GMR_TRACKER = "ckpts/g1_pingpong_tele_gmr/tracker.pt"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_motion_dataset(
  data_root: Path = DATA_ROOT,
  *,
  expected_retarget_method: str | None = None,
) -> tuple[str, ...]:
  """Reject partial or cross-robot motion sets before task construction."""
  manifest_path = data_root / "manifest.json"
  data_dir = data_root / "motions"
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("num_clips") != 43:
    raise ValueError("G1 ping-pong requires exactly 43 accepted motion clips.")
  if (
    expected_retarget_method is not None
    and manifest.get("source_retarget_method") != expected_retarget_method
  ):
    raise ValueError(
      f"Expected {expected_retarget_method!r} motions in {data_root}, got "
      f"{manifest.get('source_retarget_method')!r}."
    )
  if tuple(manifest.get("joint_names", ())) != G1_PINGPONG_JOINT_NAMES:
    raise ValueError("G1 motion joint schema differs from the 29-DoF robot.")
  expected_bodies = tuple(manifest.get("body_names", ()))
  if expected_bodies != G1_PINGPONG_BODY_NAMES:
    raise ValueError("G1 motion body schema differs from the 29-DoF robot.")
  files = []
  for record in manifest["clips"]:
    path = data_dir / record["file"]
    if not path.is_file() or _sha256(path) != record["output_sha256"]:
      raise ValueError(f"G1 prepared motion checksum mismatch: {path}")
    files.append(str(path.resolve()))
  if len(set(files)) != 43:
    raise ValueError("G1 motion manifest contains duplicate clips.")
  return tuple(files)


def _load_optional_motion_dataset(
  data_root: Path, *, expected_retarget_method: str
) -> tuple[str, ...]:
  """Load a local motion set when installed, without making it a Git dependency."""
  manifest = data_root / "manifest.json"
  motion_dir = data_root / "motions"
  if not manifest.is_file() or not motion_dir.is_dir():
    return ()
  return validate_motion_dataset(
    data_root, expected_retarget_method=expected_retarget_method
  )


MOTION_FILES = _load_optional_motion_dataset(
  DATA_ROOT, expected_retarget_method="bvh_mink_ik"
)
TELE_GMR_MOTION_FILES = _load_optional_motion_dataset(
  TELE_GMR_DATA_ROOT, expected_retarget_method="tele_gmr_football"
)


def _specialize_robot(cfg, motion_files: tuple[str, ...] = MOTION_FILES):
  """Replace every A3-dependent contract before an environment is built."""
  cfg.scene.entities["robot"] = get_g1_pingpong_robot_cfg()
  ordered_joints = SceneEntityCfg(
    "robot", joint_names=G1_PINGPONG_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for term_name in ("joint_pos", "joint_vel"):
      if term_name in group.terms:
        group.terms[term_name].params["asset_cfg"] = ordered_joints

  action = cfg.actions["joint_pos"]
  action.actuator_names = G1_PINGPONG_JOINT_NAMES
  action.scale = G1_PINGPONG_ACTION_SCALE

  motion = cfg.commands["motion"]
  motion.motion_file = ""
  motion.motion_files = motion_files
  motion.motion_directory = ""
  motion.joint_names = G1_PINGPONG_JOINT_NAMES
  motion.body_names = G1_PINGPONG_TRACKED_BODIES
  motion.anchor_body_name = G1_PINGPONG_ANCHOR_BODY

  cfg.rewards["motion_hit_arm"].params.update(
    body_names=("right_elbow_link", "right_wrist_yaw_link"),
    strike_phase=0.46,
  )
  cfg.terminations["feet_height"].params["body_names"] = G1_PINGPONG_FEET_BODIES
  cfg.viewer.body_name = G1_PINGPONG_ANCHOR_BODY
  return cfg


def g1_tracking_env_cfg(play: bool = False):
  """Random-speed, failure-adaptive tracking over all 43 G1 motions."""
  return _specialize_robot(a3_tracking_env_cfg(play))


def g1_tele_gmr_tracking_env_cfg(play: bool = False):
  """The identical Stage-1 task driven by the 43 Tele-GMR/Football clips."""
  return _specialize_robot(
    a3_tracking_env_cfg(play), motion_files=TELE_GMR_MOTION_FILES
  )


def g1_receiving_env_cfg(
  play: bool = False,
  motion_files: tuple[str, ...] = MOTION_FILES,
  tracker_file: str = "ckpts/g1_pingpong/tracker.pt",
):
  """Frozen G1 tracker with learned joint residuals and motion speed."""
  cfg = _specialize_robot(a3_receiving_env_cfg(play), motion_files=motion_files)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, FrozenTrackerActionCfg)
  action.tracker_file = tracker_file

  ball = cfg.commands["ball"]
  assert isinstance(ball, ReceivingBallCommandCfg)
  ball.racket_site_name = G1_PINGPONG_RACKET_SITE
  ball.racket_body_name = G1_PINGPONG_WRIST_BODY
  ball.racket_offset = G1_PINGPONG_MOUNT_OFFSET
  assert G1_PINGPONG_RACKET_NORMAL == (0.0, 1.0, 0.0)
  ball.racket_normal_axis = 1
  # Several G1 backhand references strike below table height. A steeper final
  # descent keeps the airborne feeder clear of the net/table before reaching
  # all 43 annotated racket poses; the A3 default makes five clips impossible.
  ball.incoming_vertical_speed = -3.5
  cfg.viewer.lookat = (1.2, 0.0, 0.8)
  return cfg


def g1_tele_gmr_receiving_env_cfg(play: bool = False):
  """Stage-2 receiving with Tele-GMR motions and their frozen Stage-1 tracker."""
  return g1_receiving_env_cfg(
    play,
    motion_files=TELE_GMR_MOTION_FILES,
    tracker_file=TELE_GMR_TRACKER,
  )


def g1_tele_gmr_hard_receiving_env_cfg(play: bool = False):
  """Hard receiving distribution for wider spatial, temporal and speed variation."""
  cfg = g1_tele_gmr_receiving_env_cfg(play)
  ball = cfg.commands["ball"]
  assert isinstance(ball, ReceivingBallCommandCfg)
  # Wider intercept and landing regions expose the high-level policy to
  # substantially more paddle/ball geometries than the baseline task.
  ball.intercept_jitter = (0.12, 0.25, 0.08)
  ball.target_jitter = (0.30, 0.45)
  ball.arrival_time_jitter = 0.18
  ball.minimum_intercept_time = 0.25
  ball.flight_time_range = (0.30, 0.70)
  ball.incoming_speed_range = (2.5, 5.5)
  ball.incoming_lateral_speed_range = (-1.5, 1.5)
  # The run warm-starts from baseline iteration 29999. Ramp the distribution
  # over the first 5000 hard-training iterations to avoid an abrupt collapse.
  ball.difficulty_ramp_start_iteration = 30000
  ball.difficulty_ramp_steps = 5000

  action = cfg.actions["joint_pos"]
  assert isinstance(action, FrozenTrackerActionCfg)
  # Give adaptive speed a wider but still causal range. The high-level action
  # now selects 0.35x--2.5x reference advancement for hard cases.
  action.min_speed = 0.35
  action.max_speed = 2.5
  return cfg


def g1_runner_cfg(stage: int = 1, tele_gmr: bool = False, hard: bool = False):
  cfg = a3_runner_cfg(stage)
  cfg.experiment_name = (
    "g1_pingpong_tracking_stage1_random_dt"
    if stage == 1
    else (
      "g1_pingpong_receive_stage2_tele_gmr_hard"
      if hard
      else "g1_pingpong_receive_stage2_tele_gmr"
      if tele_gmr
      else "g1_pingpong_receive_stage2"
    )
  )
  return cfg


def register_tasks() -> None:
  if not MOTION_FILES or not TELE_GMR_MOTION_FILES:
    return
  register_mjlab_task(
    task_id=STAGE1_TASK,
    env_cfg=g1_tracking_env_cfg(),
    play_env_cfg=g1_tracking_env_cfg(play=True),
    rl_cfg=g1_runner_cfg(),
    runner_cls=PingPongOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=TELE_GMR_STAGE1_TASK,
    env_cfg=g1_tele_gmr_tracking_env_cfg(),
    play_env_cfg=g1_tele_gmr_tracking_env_cfg(play=True),
    rl_cfg=g1_runner_cfg(),
    runner_cls=PingPongOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=STAGE2_TASK,
    env_cfg=g1_receiving_env_cfg(),
    play_env_cfg=g1_receiving_env_cfg(play=True),
    rl_cfg=g1_runner_cfg(stage=2),
    runner_cls=PingPongOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=TELE_GMR_STAGE2_TASK,
    env_cfg=g1_tele_gmr_receiving_env_cfg(),
    play_env_cfg=g1_tele_gmr_receiving_env_cfg(play=True),
    rl_cfg=g1_runner_cfg(stage=2, tele_gmr=True),
    runner_cls=PingPongOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=TELE_GMR_HARD_STAGE2_TASK,
    env_cfg=g1_tele_gmr_hard_receiving_env_cfg(),
    play_env_cfg=g1_tele_gmr_hard_receiving_env_cfg(play=True),
    rl_cfg=g1_runner_cfg(stage=2, tele_gmr=True, hard=True),
    runner_cls=PingPongOnPolicyRunner,
  )

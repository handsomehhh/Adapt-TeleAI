"""Training-compatible math, motion interpolation and hierarchical control."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACKER_TERMS = {
  "command": 29,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "projected_gravity": 3,
  "base_ang_vel": 3,
  "joint_pos": 29,
  "joint_vel": 29,
  "actions": 29,
  "motion_dt_sample": 1,
}
ACTOR_TERMS = {**TRACKER_TERMS, "ball": 23, "phase": 1, "high_level_action": 30}


def rotation(q):
  q = np.asarray(q, dtype=np.float64)
  if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
    raise ValueError("Expected a finite nonzero WXYZ quaternion")
  w, x, y, z = q / np.linalg.norm(q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def quaternion_multiply(q, p):
  w, v = q[0], np.asarray(q[1:])
  s, u = p[0], np.asarray(p[1:])
  return np.r_[w * s - v @ u, w * u + s * v + np.cross(v, u)]


def slerp(q0, q1, fraction):
  q0, q1 = np.asarray(q0), np.asarray(q1)
  q0 = q0 / np.linalg.norm(q0)
  q1 = q1 / np.linalg.norm(q1)
  dot = float(np.dot(q0, q1))
  if dot < 0:
    q1, dot = -q1, -dot
  dot = min(dot, 1.0)
  if dot > 0.9995:
    q = q0 + fraction * (q1 - q0)
    return q / np.linalg.norm(q)
  angle = np.arccos(dot)
  return (np.sin((1 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1) / np.sin(
    angle
  )


@dataclass
class RobotState:
  q: np.ndarray
  dq: np.ndarray
  root_pos: np.ndarray
  root_quat: np.ndarray
  gyro: np.ndarray
  anchor_pos: np.ndarray
  anchor_quat: np.ndarray
  racket_pos: np.ndarray
  racket_normal: np.ndarray
  racket_velocity: np.ndarray | None = None
  racket_angular_velocity: np.ndarray | None = None


@dataclass
class BallState:
  position: np.ndarray
  velocity: np.ndarray
  intercept: np.ndarray
  remaining: float
  target: np.ndarray
  hit: bool = False


class Motion:
  def __init__(self, path: Path, joint_names: list[str], anchor: str):
    with np.load(path, allow_pickle=False) as data:
      names = list(data["joint_names"].astype(str))
      bodies = list(data["body_names"].astype(str))
      cols = [names.index(n) for n in joint_names]
      self.q = data["joint_pos"][:, cols].copy()
      self.dq = data["joint_vel"][:, cols].copy()
      self.body_pos = data["body_pos_w"].copy()
      self.body_quat = data["body_quat_w"].copy()
      self.body_lin_vel = (
        data["body_lin_vel_w"].copy()
        if "body_lin_vel_w" in data
        else np.zeros_like(self.body_pos)
      )
      self.body_ang_vel = (
        data["body_ang_vel_w"].copy()
        if "body_ang_vel_w" in data
        else np.zeros_like(self.body_pos)
      )
      self.body_names = bodies
      self.anchor = bodies.index(anchor)
      self.root = bodies.index("pelvis")
      self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      if not np.isfinite(self.fps) or self.fps <= 0:
        raise ValueError("Invalid motion fps")
      self.annotated_strike = None
      for key in ("strike_time_s", "strike_frame", "hope_strike_frame"):
        if key in data:
          value = float(np.asarray(data[key]).reshape(-1)[0])
          self.annotated_strike = value if key == "strike_time_s" else value / self.fps
          break
    if len(self.q) < 2:
      raise ValueError("Invalid motion timing")
    frames = len(self.q)
    if len(names) != len(set(names)) or len(bodies) != len(set(bodies)):
      raise ValueError("Duplicate motion joint/body names")
    if self.q.shape != (frames, 29) or self.dq.shape != self.q.shape:
      raise ValueError("Motion joint arrays must have shape (frames,29)")
    if self.body_pos.shape != (frames, len(bodies), 3) or self.body_quat.shape != (
      frames,
      len(bodies),
      4,
    ):
      raise ValueError("Motion body array shapes differ from names")
    if (
      self.body_lin_vel.shape != self.body_pos.shape
      or self.body_ang_vel.shape != self.body_pos.shape
    ):
      raise ValueError("Motion body velocity shapes differ")
    if not all(
      np.isfinite(a).all()
      for a in (
        self.q,
        self.dq,
        self.body_pos,
        self.body_quat,
        self.body_lin_vel,
        self.body_ang_vel,
      )
    ):
      raise ValueError("Nonfinite motion")
    self.duration = (len(self.q) - 1) / self.fps
    if np.any(np.linalg.norm(self.body_quat, axis=-1) < 1e-8):
      raise ValueError("Zero motion quaternion")
    if (
      self.annotated_strike is not None
      and not 0 <= self.annotated_strike <= self.duration
    ):
      raise ValueError("Invalid annotated strike time")

  def body_position(self, name, t):
    frame = np.clip(t * self.fps, 0, len(self.q) - 1)
    i, f = int(frame), frame - int(frame)
    j = min(i + 1, len(self.q) - 1)
    body = self.body_names.index(name)
    return self.body_pos[i, body] * (1 - f) + self.body_pos[j, body] * f

  def sample(self, t):
    frame = np.clip(t * self.fps, 0, len(self.q) - 1)
    i = int(frame)
    j = min(i + 1, len(self.q) - 1)
    f = frame - i
    return (
      self.q[i] * (1 - f) + self.q[j] * f,
      self.body_pos[i, self.anchor] * (1 - f) + self.body_pos[j, self.anchor] * f,
      slerp(self.body_quat[i, self.anchor], self.body_quat[j, self.anchor], f),
    )

  def strike_position(self, cfg):
    # Training rounds the annotated hit time to a source frame.
    i = min(max(round(self.strike_time(cfg) * self.fps), 0), len(self.q) - 1)
    body = self.body_names.index(cfg["racket_body_name"])
    return (
      self.body_pos[i, body] + rotation(self.body_quat[i, body]) @ cfg["racket_offset"]
    )

  def strike_time(self, cfg):
    if self.annotated_strike is not None:
      return self.annotated_strike
    return round((len(self.q) - 1) * cfg["strike_phase"]) / self.fps


def ball_observation(state: RobotState, ball: BallState, cfg):
  inv = rotation(state.root_quat).T
  parts = [
    inv @ (ball.position - state.root_pos),
    inv @ ball.velocity,
    inv @ (ball.intercept - state.root_pos),
    [max(ball.remaining, 0)],
    inv @ (ball.target - state.root_pos),
  ]
  acceleration = np.array([0.0, 0.0, -cfg["gravity"]])
  acceleration -= cfg["drag"] * np.linalg.norm(ball.velocity) * ball.velocity
  for t in cfg["prediction_times"]:
    predicted = ball.position + ball.velocity * t + 0.5 * acceleration * t * t
    parts.append(inv @ (predicted - state.root_pos))
  parts.append([float(ball.hit)])
  return np.concatenate(parts).astype(np.float32)


class JitPolicy:
  def __init__(self, path, input_dim, output_dim):
    import torch

    self.model = torch.jit.load(str(path), map_location="cpu").eval()
    self.input_dim, self.output_dim = input_dim, output_dim
    self(np.zeros(input_dim, dtype=np.float32))

  def __call__(self, obs):
    import torch

    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape != (self.input_dim,) or not np.isfinite(obs).all():
      raise ValueError("Policy observation shape/values invalid")
    with torch.inference_mode():
      result = self.model(torch.from_numpy(obs[None])).cpu().numpy()[0]
    if result.shape != (self.output_dim,) or not np.isfinite(result).all():
      raise ValueError("Policy action shape/values invalid")
    return result


class Controller:
  def __init__(self, manifest, motion, tracker, actor=None):
    self.cfg, self.motion = manifest, motion
    self.tracker, self.actor = tracker, actor
    self.dt = manifest["control_dt"]
    self.default = np.asarray(manifest["robot"]["default_q"])
    self.scale = np.asarray(manifest["robot"]["action_scale"])
    self.residual_scales = np.asarray(manifest["action"]["residual_scales"])
    self.reset()

  def reset(self):
    self.t = 0.0
    self.delta_dt = 0.0
    self.previous_joint = np.zeros(29, dtype=np.float32)
    self.previous_high = np.zeros(30, dtype=np.float32)
    self.debug = {}

  def start_from(self, previous):
    """Start a new swing without losing the tracker's preceding action history."""
    self.reset()
    self.previous_joint = previous.previous_joint.copy()
    self.delta_dt = previous.delta_dt

  def observation(self, state):
    q, pos, quat = self.motion.sample(self.t)
    inv = rotation(state.anchor_quat).T
    terms = [
      q,
      inv @ (pos - state.anchor_pos),
      (inv @ rotation(quat))[:, :2].reshape(-1),
      rotation(state.root_quat).T @ [0.0, 0.0, -1.0],
      state.gyro,
      state.q - self.default,
      state.dq,
      self.previous_joint,
      [self.delta_dt],
    ]
    return np.concatenate(terms).astype(np.float32)

  def step(
    self, state, ball=None, *, fixed_speed=None, zero_residual=False, tracker_only=False
  ):
    tracker_obs = self.observation(state)
    if self.actor is not None and not tracker_only:
      if ball is None:
        raise ValueError("Receiving requires a valid ball state and receive plan")
      actor_obs = np.concatenate(
        [
          tracker_obs,
          ball_observation(state, ball, self.cfg["ball"]),
          [min(self.t / self.motion.duration, 1)],
          self.previous_high,
        ]
      ).astype(np.float32)
      high = np.asarray(self.actor(actor_obs), dtype=np.float32)
      if high.shape != (30,) or not np.isfinite(high).all():
        raise ValueError("Invalid high-level action")
      u = float(np.clip(high[-1], -1, 1))
      action_cfg = self.cfg["action"]
      speed = 1 + u * (
        action_cfg["max_speed"] - 1 if u >= 0 else 1 - action_cfg["min_speed"]
      )
      residual = np.clip(
        high[:-1], -action_cfg["residual_clip"], action_cfg["residual_clip"]
      )
      if zero_residual:
        residual = np.zeros(29)
    else:
      actor_obs, high = np.array([]), np.zeros(30, dtype=np.float32)
      speed, residual = 1.0, np.zeros(29)
    if fixed_speed is not None:
      if not np.isfinite(fixed_speed) or fixed_speed <= 0:
        raise ValueError("fixed_speed must be finite and positive")
      speed = fixed_speed
    base = np.asarray(self.tracker(tracker_obs), dtype=np.float32)
    if base.shape != (29,) or not np.isfinite(base).all():
      raise ValueError("Invalid tracker action")
    joint = base + self.residual_scales * residual
    target = self.default + self.scale * joint
    if not np.isfinite(target).all():
      raise ValueError("Nonfinite joint command")
    self.debug = {
      "reference_time": self.t,
      "reference_delta": self.delta_dt,
      "tracker_obs": tracker_obs,
      "actor_obs": actor_obs,
      "base_action": base,
      "high_action": high,
      "joint_action": joint,
      "target_q": target,
      "speed": speed,
    }
    self.previous_joint = joint.astype(np.float32)
    self.previous_high = high.copy()
    self.delta_dt = self.dt * (speed - 1)
    # The timeline continues through warmdown; only reference interpolation
    # and phase are clamped. This matches MotionCommand in the training env.
    self.t += self.dt * speed
    return target

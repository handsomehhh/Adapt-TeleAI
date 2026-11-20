"""Independent legal feeds, analytic ball dynamics and measured-state prediction.

Ball state is analytic; contact uses the *executed* robot paddle, never MVAE FK.
The predictor sees noisy/delayed position packets, not simulator velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply


@dataclass
class BallPhysics:
  gravity: float = 9.81
  drag: float = 0.12
  table_height: float = 0.76
  table_near: float = 0.65
  table_far: float = 3.39
  table_half_width: float = 0.7625
  net_x: float = 2.02
  net_height: float = 0.1525
  radius: float = 0.02
  table_restitution: float = 0.85
  horizontal_retention: float = 0.75
  paddle_restitution: float = 0.88
  paddle_radius: float = 0.08


def integrate(position, velocity, dt, physics):
  acceleration = -physics.drag * velocity.norm(dim=-1, keepdim=True) * velocity
  acceleration = acceleration.clone()
  acceleration[..., 2] -= physics.gravity
  next_velocity = velocity + acceleration * dt
  return position + (velocity + next_velocity) * (0.5 * dt), next_velocity


def table_bounce(previous, position, velocity, physics):
  height = physics.table_height + physics.radius
  fraction = (
    (previous[..., 2] - height) / (previous[..., 2] - position[..., 2]).clamp_min(1e-8)
  ).clamp(0, 1)
  point = previous + fraction[..., None] * (position - previous)
  hit = (
    (previous[..., 2] >= height)
    & (position[..., 2] < height)
    & (velocity[..., 2] < 0)
    & (point[..., 0] >= physics.table_near)
    & (point[..., 0] <= physics.table_far)
    & (point[..., 1].abs() <= physics.table_half_width)
  )
  position = position.clone()
  velocity = velocity.clone()
  position[..., 2] = torch.where(hit, 2 * height - position[..., 2], position[..., 2])
  velocity[..., :2] *= torch.where(hit, physics.horizontal_retention, 1.0)[..., None]
  velocity[..., 2] = torch.where(
    hit, -physics.table_restitution * velocity[..., 2], velocity[..., 2]
  )
  return position, velocity, hit, point


def swept_paddle(
  previous_ball, ball, velocity, previous_paddle, paddle, normal, dt, physics
):
  old = ((previous_ball - previous_paddle) * normal).sum(-1)
  new = ((ball - paddle) * normal).sum(-1)
  side = torch.where(old >= 0, 1.0, -1.0)
  surface = side * physics.radius
  fraction = (
    (old - surface) / (old - new).where((old - new).abs() > 1e-8, torch.ones_like(old))
  ).clamp(0, 1)
  relative = (previous_ball - previous_paddle) + fraction[:, None] * (
    (ball - paddle) - (previous_ball - previous_paddle)
  )
  radial = relative - (relative * normal).sum(-1, keepdim=True) * normal
  paddle_velocity = (paddle - previous_paddle) / dt
  approach = ((velocity - paddle_velocity) * normal).sum(-1)
  hit = (
    (side * old >= physics.radius)
    & (side * new <= physics.radius)
    & (radial.norm(dim=-1) <= physics.paddle_radius)
    & (side * approach < -1e-4)
  )
  reflected = velocity - (1 + physics.paddle_restitution) * approach[:, None] * normal
  contact_ball = previous_ball + fraction[:, None] * (ball - previous_ball)
  outgoing = contact_ball + reflected * ((1 - fraction) * dt)[:, None]
  return (
    hit,
    torch.where(hit[:, None], outgoing, ball),
    torch.where(hit[:, None], reflected, velocity),
  )


def predict_trajectory(position, velocity, physics, count=30, dt=0.02):
  points = []
  for _ in range(count):
    old = position
    position, velocity = integrate(position, velocity, dt, physics)
    position, velocity, _, _ = table_bounce(old, position, velocity, physics)
    points.append(position)
  return torch.stack(points, 1)


def legal_feed_bank(physics, size=1024, seed=42, lateral=0.45):
  """Rejection sampling only on ball flight geometry; independent of motion data."""
  rng = np.random.default_rng(seed)
  accepted = []
  for _ in range(16):
    count = max(size * 4, 2048)
    start = np.stack(
      (
        rng.uniform(2.6, 3.25, count),
        rng.uniform(-lateral, lateral, count),
        rng.uniform(1.0, 1.55, count),
      ),
      -1,
    )
    speed = np.stack(
      (
        rng.uniform(-6.5, -3.5, count),
        rng.uniform(-0.35, 0.35, count),
        rng.uniform(-1.0, 1.0, count),
      ),
      -1,
    )
    p, v = start.copy(), speed.copy()
    bounces = np.zeros(count, dtype=np.int64)
    net_ok = np.zeros(count, dtype=bool)
    bad = np.zeros(count, dtype=bool)
    done = np.zeros(count, dtype=bool)
    for _step in range(180):
      old = p.copy()
      acceleration = -physics.drag * np.linalg.norm(v, axis=-1, keepdims=True) * v
      acceleration[:, 2] -= physics.gravity
      new_v = v + acceleration * 0.01
      p += (v + new_v) * 0.005
      v = new_v
      crossing = (old[:, 0] > physics.net_x) & (p[:, 0] <= physics.net_x)
      t = ((old[:, 0] - physics.net_x) / np.maximum(old[:, 0] - p[:, 0], 1e-8)).clip(
        0, 1
      )
      net_z = old[:, 2] + t * (p[:, 2] - old[:, 2])
      net_ok |= crossing & (
        net_z > physics.table_height + physics.net_height + physics.radius
      )
      bad |= crossing & ~net_ok
      height = physics.table_height + physics.radius
      fraction = ((old[:, 2] - height) / np.maximum(old[:, 2] - p[:, 2], 1e-8)).clip(
        0, 1
      )
      point = old + fraction[:, None] * (p - old)
      bounce = (
        (old[:, 2] >= height)
        & (p[:, 2] < height)
        & (v[:, 2] < 0)
        & (point[:, 0] >= physics.table_near)
        & (point[:, 0] <= physics.table_far)
        & (abs(point[:, 1]) <= physics.table_half_width)
      )
      bad |= bounce & (point[:, 0] >= physics.net_x)
      bounces += bounce
      p[bounce, 2] = 2 * height - p[bounce, 2]
      v[bounce, 2] *= -physics.table_restitution
      v[bounce, :2] *= physics.horizontal_retention
      plane = (old[:, 0] > 0.2) & (p[:, 0] <= 0.2) & ~done
      good = (
        plane
        & net_ok
        & ~bad
        & (bounces == 1)
        & (p[:, 2] >= 0.65)
        & (p[:, 2] <= 1.35)
        & (abs(p[:, 1]) <= lateral)
      )
      accepted.extend(np.concatenate((start[good], speed[good]), -1))
      done |= plane
    if len(accepted) >= size:
      return np.asarray(accepted[:size], dtype=np.float32)
  raise RuntimeError(
    f"Could only sample {len(accepted)} legal feeds; inspect ball parameters"
  )


class PositionEstimator:
  """Timestamped least-squares velocity from received positions only."""

  def __init__(self, count, device, window=5):
    self.positions = torch.zeros(count, window, 3, device=device)
    self.times = torch.zeros(count, window, device=device)
    self.count = torch.zeros(count, dtype=torch.long, device=device)

  def reset(self, ids, position, time):
    self.positions[ids] = position[:, None]
    self.times[ids] = time[:, None]
    self.count[ids] = 0

  def observe(self, position, time, received):
    received = (
      received
      & torch.isfinite(position).all(-1)
      & torch.isfinite(time)
      & ((self.count == 0) | (time > self.times[:, -1] + 1e-6))
    )
    self.positions[received] = torch.cat(
      (self.positions[received, 1:], position[received, None]), 1
    )
    self.times[received] = torch.cat(
      (self.times[received, 1:], time[received, None]), 1
    )
    self.count[received] += 1

  def estimate(self, now, gravity=9.81):
    mask = (
      torch.arange(self.times.shape[1], device=now.device)[None]
      >= (self.times.shape[1] - self.count.clamp(max=self.times.shape[1]))[:, None]
    )
    weight = mask.float()
    n = weight.sum(-1).clamp_min(1)
    mean_t = (self.times * weight).sum(-1) / n
    mean_p = (self.positions * weight[..., None]).sum(1) / n[:, None]
    delta = (self.times - mean_t[:, None]) * weight
    velocity = (delta[..., None] * (self.positions - mean_p[:, None])).sum(
      1
    ) / delta.square().sum(-1).clamp_min(1e-6)[:, None]
    age = (now - self.times[:, -1]).clamp_min(0)
    current_velocity = velocity.clone()
    current_velocity[:, 2] -= gravity * (now - mean_t)
    position = self.positions[:, -1] + velocity * age[:, None]
    position[:, 2] -= gravity * (
      0.5 * age.square() + age * (self.times[:, -1] - mean_t)
    )
    valid = (self.count >= 2) & (age < 0.15)
    return position, current_velocity, valid, age


@dataclass(kw_only=True)
class RallyBallCfg(CommandTermCfg):
  physics: BallPhysics = field(default_factory=BallPhysics)
  entity_name: str = "robot"
  active: bool = True
  prediction_steps: int = 30
  noise: float = 0.005
  dropout: float = 0.05
  delay_steps: int = 2
  feed_lateral: float = 0.45
  max_time: float = 2.5
  freeze_on_done: bool = True
  """Training freezes scored balls; continuous deployment may show their aftermath."""

  def build(self, env):
    return RallyBall(self, env)


class RallyBall(CommandTerm):
  cfg: RallyBallCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.wrist = self.robot.body_names.index("right_wrist_yaw_link")
    self.p = torch.zeros(self.num_envs, 3, device=self.device)
    self.v = torch.zeros_like(self.p)
    self.target = torch.zeros_like(self.p)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.cooldown = torch.zeros_like(self.elapsed)
    self.hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.net_clear = torch.zeros_like(self.hit)
    self.success = torch.zeros_like(self.hit)
    self.done = torch.zeros_like(self.hit)
    self.double_hit = torch.zeros_like(self.hit)
    self.hit_event = torch.zeros_like(self.hit)
    self.net_event = torch.zeros_like(self.hit)
    self.incoming_bounces = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.illegal_contact = torch.zeros_like(self.hit)
    self.landing_score = torch.zeros_like(self.elapsed)
    self.min_distance = torch.full_like(self.elapsed, 10)
    self.last_result = {
      name: torch.zeros_like(self.elapsed)
      for name in (
        "hit",
        "net_clear",
        "success",
        "double_hit",
        "landing_score",
        "min_distance",
        "elapsed",
        "illegal_contact",
      )
    }
    self.last_result["fallen"] = torch.zeros_like(self.elapsed)
    self.estimator = PositionEstimator(self.num_envs, self.device)
    self.position_history = torch.zeros(
      self.num_envs, cfg.delay_steps + 1, 3, device=self.device
    )
    self.observed_p = torch.zeros_like(self.p)
    self.observed_v = torch.zeros_like(self.p)
    self.observed_valid = torch.zeros_like(self.hit)
    self.observed_age = torch.zeros_like(self.elapsed)
    self.future = torch.zeros(
      self.num_envs, cfg.prediction_steps, 3, device=self.device
    )
    bank = (
      legal_feed_bank(
        cfg.physics, seed=int(env.cfg.seed or 42), lateral=cfg.feed_lateral
      )
      if cfg.active
      else np.zeros((1, 6), dtype=np.float32)
    )
    self.bank = torch.as_tensor(bank, device=self.device)
    self.previous_paddle, self.previous_normal = self.paddle()
    for name in ("hit", "net_clear", "success", "double_hit", "landing_score"):
      self.metrics[name] = torch.zeros_like(self.elapsed)

  @property
  def command(self):
    base = self.robot.data.root_link_pos_w
    relative = torch.cat(
      (
        self.observed_p - base,
        self.observed_v,
        (self.future - base[:, None]).flatten(1),
        self.target - base,
        self.observed_valid.float()[:, None],
        self.observed_age[:, None],
      ),
      -1,
    )
    return torch.where(
      self.observed_valid[:, None], relative, torch.zeros_like(relative)
    )

  def paddle(self):
    from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
      G1_PINGPONG_MOUNT_OFFSET,
    )

    quat = self.robot.data.body_link_quat_w[:, self.wrist]
    center = self.robot.data.body_link_pos_w[:, self.wrist] + quat_apply(
      quat,
      torch.tensor(G1_PINGPONG_MOUNT_OFFSET, device=self.device).expand(
        self.num_envs, -1
      ),
    )
    normal = matrix_from_quat(quat)[:, :, 1]
    return center.clone(), normal.clone()

  def _resample_command(self, env_ids):
    ids = env_ids
    for name, value in self.last_result.items():
      if name == "fallen":
        value[ids] = (
          (self.robot.data.root_link_pos_w[ids, 2] < 0.45)
          | (self.robot.data.projected_gravity_b[ids, 2] > -0.5)
        ).float()
      else:
        value[ids] = getattr(self, name)[ids].float()
    sample = self.bank[torch.randint(len(self.bank), (len(ids),), device=self.device)]
    self.p[ids] = sample[:, :3] + self._env.scene.env_origins[ids]
    self.v[ids] = sample[:, 3:]
    self.target[ids] = (
      torch.tensor(
        [2.7, 0, self.cfg.physics.table_height + self.cfg.physics.radius],
        device=self.device,
      )
      + self._env.scene.env_origins[ids]
    )
    self.elapsed[ids] = 0
    self.cooldown[ids] = 0
    for buffer in (
      self.hit,
      self.net_clear,
      self.success,
      self.done,
      self.double_hit,
      self.hit_event,
      self.net_event,
      self.illegal_contact,
    ):
      buffer[ids] = False
    self.landing_score[ids] = 0
    self.incoming_bounces[ids] = 0
    self.min_distance[ids] = 10
    self.position_history[ids] = self.p[ids, None]
    self.estimator.reset(ids, self.p[ids], self.elapsed[ids])
    self.observed_valid[ids] = False
    self.observed_age[ids] = 0
    paddle, normal = self.paddle()
    self.previous_paddle[ids], self.previous_normal[ids] = paddle[ids], normal[ids]

  def compute_substep(self, dt):
    if not self.cfg.active:
      return
    physics = self.cfg.physics
    scoring = ~self.done if not self.cfg.freeze_on_done else None
    origin = self._env.scene.env_origins
    old = self.p.clone()
    p, v = integrate(self.p, self.v, dt, physics)
    paddle, normal = self.paddle()
    averaged = torch.nn.functional.normalize(normal + self.previous_normal, dim=-1)
    contact, p, v = swept_paddle(
      old, p, v, self.previous_paddle, paddle, averaged, dt, physics
    )
    contact &= ~self.done & (self.cooldown <= 0)
    illegal = contact & ~self.hit & (self.incoming_bounces != 1)
    self.illegal_contact |= illegal
    contact &= ~illegal
    # Apply collision only for active contacts, including post-hit double touches.
    free_p, free_v = integrate(self.p, self.v, dt, physics)
    p, v = (
      torch.where(contact[:, None], p, free_p),
      torch.where(contact[:, None], v, free_v),
    )
    self.double_hit |= contact & self.hit
    self.hit_event |= contact & ~self.hit
    self.hit |= contact
    self.cooldown = torch.where(contact, 0.03, (self.cooldown - dt).clamp_min(0))
    local_old, local = old - origin, p - origin
    p_local, v, bounce, point = table_bounce(local_old, local, v, physics)
    new_bounces = bounce & ~self.hit
    if scoring is not None:
      new_bounces &= scoring
    self.incoming_bounces += new_bounces.long()
    p = p_local + origin
    crossing = (
      (local_old[:, 0] - physics.net_x) * (local[:, 0] - physics.net_x) <= 0
    ) & ((local[:, 0] - local_old[:, 0]).abs() > 1e-7)
    fraction = (
      (physics.net_x - local_old[:, 0])
      / (local[:, 0] - local_old[:, 0]).where(
        (local[:, 0] - local_old[:, 0]).abs() > 1e-7, torch.ones_like(self.elapsed)
      )
    ).clamp(0, 1)
    net_z = local_old[:, 2] + fraction * (local[:, 2] - local_old[:, 2])
    net_blocked = (
      crossing
      & (net_z <= physics.table_height + physics.net_height + physics.radius)
      & (p_local[:, 1].abs() <= physics.table_half_width)
    )
    cleared = crossing & self.hit & (v[:, 0] > 0) & ~net_blocked
    if scoring is not None:
      cleared &= scoring
    self.net_event |= cleared & ~self.net_clear
    self.net_clear |= cleared
    landed = bounce & self.hit & ~self.done
    valid_landing = (
      landed & self.net_clear & (point[:, 0] > physics.net_x) & ~self.double_hit
    )
    error = (point[:, :2] + origin[:, :2] - self.target[:, :2]).square().sum(-1)
    self.landing_score = torch.where(
      valid_landing, torch.exp(-error / 0.25), self.landing_score
    )
    self.success |= valid_landing
    self.done |= (
      illegal
      | (self.incoming_bounces > 1)
      | landed
      | net_blocked
      | (p_local[:, 0] < -0.7)
      | (p_local[:, 2] < physics.radius)
      | (self.elapsed > self.cfg.max_time)
    )
    if self.cfg.freeze_on_done:
      self.p = torch.where(self.done[:, None], self.p, p)
      self.v = torch.where(self.done[:, None], torch.zeros_like(v), v)
    else:
      self.p, self.v = p, v
    self.elapsed += dt
    self.previous_paddle, self.previous_normal = paddle, normal
    minimum = torch.minimum(self.min_distance, (p - paddle).norm(dim=-1))
    self.min_distance = (
      minimum if scoring is None else torch.where(scoring, minimum, self.min_distance)
    )

  def _update_command(self):
    # Rewards consume events before command updates at the end of a control step.
    self.hit_event.zero_()
    self.net_event.zero_()
    self.position_history = torch.cat(
      (self.position_history[:, 1:], self.p[:, None]), 1
    )
    delay = torch.randint(
      self.cfg.delay_steps + 1, (self.num_envs,), device=self.device
    )
    index = self.cfg.delay_steps - delay
    measured = self.position_history[
      torch.arange(self.num_envs, device=self.device), index
    ]
    measured = measured + torch.randn_like(measured) * self.cfg.noise
    timestamp = self.elapsed - delay * self._env.step_dt
    received = (torch.rand(self.num_envs, device=self.device) >= self.cfg.dropout) & (
      timestamp >= 0
    )
    self.estimator.observe(measured, timestamp, received)
    self.observed_p, self.observed_v, valid, self.observed_age = (
      self.estimator.estimate(self.elapsed, self.cfg.physics.gravity)
    )
    self.observed_valid = valid & ~self.done & self.cfg.active
    self.future = (
      predict_trajectory(
        self.observed_p - self._env.scene.env_origins,
        self.observed_v,
        self.cfg.physics,
        self.cfg.prediction_steps,
        self._env.step_dt,
      )
      + self._env.scene.env_origins[:, None]
    )

  def _update_metrics(self):
    for name in self.metrics:
      self.metrics[name][:] = getattr(self, name).float()

  def _debug_vis_impl(self, visualizer):
    for index in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        self.p[index].detach().cpu().numpy(),
        radius=self.cfg.physics.radius,
        color=(1.0, 0.4, 0.0, 1.0),
      )

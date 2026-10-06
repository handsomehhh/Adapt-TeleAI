"""Analytic table-tennis ball with swept, moving-paddle contact.

The ball is integrated at physics frequency and rendered through debug geometry.
It is deliberately independent of MuJoCo's solver: the 2.7 g ball has negligible
robot reaction, while continuous collision tests avoid tunnelling through the
thin paddle. A contact changes velocity only at an intersected paddle surface;
there is no distance-triggered teleport or automatic successful return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand, _resolve_motion_paths
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def ballistic_step(
  position: torch.Tensor,
  velocity: torch.Tensor,
  dt: float | torch.Tensor,
  gravity: float = 9.81,
  drag: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Second-order position update with quadratic drag acceleration."""
  if isinstance(dt, torch.Tensor) and dt.ndim == 1:
    dt = dt.unsqueeze(-1)
  acceleration = -drag * velocity.norm(dim=-1, keepdim=True) * velocity
  acceleration = acceleration.clone()
  acceleration[..., 2] -= gravity
  return (
    position + velocity * dt + 0.5 * acceleration * dt**2,
    velocity + acceleration * dt,
  )


def swept_racket_collision(
  position0: torch.Tensor,
  position1: torch.Tensor,
  velocity: torch.Tensor,
  center0: torch.Tensor,
  center1: torch.Tensor,
  normal: torch.Tensor,
  dt: float,
  *,
  ball_radius: float = 0.02,
  racket_radius: float = 0.08,
  restitution: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Sweep a sphere against either face of a translating circular paddle.

  The face normal is the current substep orientation. Rim impacts are excluded:
  the contact projection must lie inside the actual disk, not an inflated disk.
  Restitution acts on ball velocity relative to the moving paddle.
  """
  normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
  signed0 = ((position0 - center0) * normal).sum(-1)
  signed1 = ((position1 - center1) * normal).sum(-1)
  side = torch.where(signed0 >= 0, 1.0, -1.0)
  approach = side * (signed0 - signed1)
  fraction = ((signed0.abs() - ball_radius) / approach.clamp_min(1e-8)).clamp(0, 1)
  intersection = position0 + fraction[:, None] * (position1 - position0)
  center_at_hit = center0 + fraction[:, None] * (center1 - center0)
  offset = intersection - center_at_hit
  radial = offset - (offset * normal).sum(-1, keepdim=True) * normal
  paddle_velocity = (center1 - center0) / dt
  relative_normal_speed = ((velocity - paddle_velocity) * normal).sum(-1)
  contact = (
    (approach > 1e-8)
    & (side * signed1 <= ball_radius)
    & (side * relative_normal_speed < -1e-4)
    & (radial.square().sum(-1) <= racket_radius**2)
  )
  reflected = velocity - (1.0 + restitution) * relative_normal_speed[:, None] * normal
  continued = intersection + reflected * ((1.0 - fraction) * dt)[:, None]
  return (
    torch.where(contact[:, None], continued, position1),
    torch.where(contact[:, None], reflected, velocity),
    contact,
  )


def swept_table_collision(
  position0: torch.Tensor,
  position1: torch.Tensor,
  velocity: torch.Tensor,
  dt: float,
  *,
  near_x: float = 0.65,
  far_x: float = 3.39,
  width: float = 1.525,
  height: float = 0.76,
  ball_radius: float = 0.02,
  restitution: float = 0.88,
  tangential_damping: float = 0.96,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Downward sphere/table crossing, including finite table boundaries."""
  plane = height + ball_radius
  fraction = (
    (position0[:, 2] - plane) / (position0[:, 2] - position1[:, 2]).clamp_min(1e-8)
  ).clamp(0, 1)
  intersection = position0 + fraction[:, None] * (position1 - position0)
  contact = (
    (position0[:, 2] >= plane)
    & (position1[:, 2] <= plane)
    & (velocity[:, 2] < 0)
    & (intersection[:, 0] >= near_x)
    & (intersection[:, 0] <= far_x)
    & (intersection[:, 1].abs() <= width / 2)
  )
  reflected = velocity.clone()
  reflected[:, :2] *= tangential_damping
  reflected[:, 2] *= -restitution
  continued = intersection + reflected * ((1 - fraction) * dt)[:, None]
  return (
    torch.where(contact[:, None], continued, position1),
    torch.where(contact[:, None], reflected, velocity),
    contact,
    intersection,
  )


def swept_net_collision(
  position0: torch.Tensor,
  position1: torch.Tensor,
  velocity: torch.Tensor,
  dt: float,
  *,
  net_x: float = 2.02,
  width: float = 1.525,
  table_height: float = 0.76,
  net_height: float = 0.1525,
  ball_radius: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Block crossings through the net; separately report legal forward clearance."""
  delta = position1[:, 0] - position0[:, 0]
  safe_delta = torch.where(delta.abs() > 1e-8, delta, torch.ones_like(delta))
  side = torch.where(position0[:, 0] < net_x, -1.0, 1.0)
  surface = net_x + side * ball_radius
  fraction = ((surface - position0[:, 0]) / safe_delta).clamp(0, 1)
  crossing = (
    ((position0[:, 0] - surface) * side >= 0)
    & ((position1[:, 0] - surface) * side <= 0)
    & (delta * side < 0)
  )
  intersection = position0 + fraction[:, None] * (position1 - position0)
  within_width = intersection[:, 1].abs() <= width / 2 + ball_radius
  blocked = (
    crossing
    & within_width
    & (intersection[:, 2] <= table_height + net_height + ball_radius)
    & (intersection[:, 2] >= table_height - ball_radius)
  )
  # Require the centre to cross the actual net plane before awarding clearance.
  centre_fraction = ((net_x - position0[:, 0]) / safe_delta).clamp(0, 1)
  centre_crossing = position0 + centre_fraction[:, None] * (position1 - position0)
  cleared = (
    (position0[:, 0] < net_x)
    & (position1[:, 0] >= net_x)
    & (centre_crossing[:, 2] > table_height + net_height + ball_radius)
    & (centre_crossing[:, 1].abs() <= width / 2)
  )
  reflected = velocity.clone()
  reflected[:, 0] *= -0.1
  reflected[:, 1:] *= 0.5
  continued = intersection + reflected * ((1 - fraction) * dt)[:, None]
  return (
    torch.where(blocked[:, None], continued, position1),
    torch.where(blocked[:, None], reflected, velocity),
    blocked,
    cleared & ~blocked,
  )


@dataclass(kw_only=True)
class ReceivingBallCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  motion_command_name: str = "motion"
  racket_site_name: str = "racket_center"
  racket_body_name: str = "right_wrist_yaw_Link"
  racket_offset: tuple[float, float, float] = (
    0.210211399202899,
    0.0320784994676765,
    0.0320358706296689,
  )
  racket_normal_axis: int = 1
  racket_radius: float = 0.08
  ball_radius: float = 0.02
  racket_restitution: float = 0.85
  table_restitution: float = 0.88
  table_near_x: float = 0.65
  table_far_x: float = 3.39
  table_width: float = 1.525
  table_height: float = 0.76
  net_height: float = 0.1525
  gravity: float = 9.81
  drag: float = 0.0
  strike_phase: float = 0.46
  minimum_intercept_time: float = 0.35
  arrival_time_jitter: float = 0.0
  """Uniform arrival-time offset in ± seconds; zero preserves nominal clip timing."""
  flight_time_range: tuple[float, float] = (0.4, 0.5)
  incoming_speed_range: tuple[float, float] = (3.5, 4.5)
  incoming_lateral_speed_range: tuple[float, float] = (0.0, 0.0)
  """Final Y velocity range for the incoming feed, in m/s."""
  incoming_vertical_speed: float = -2.7
  intercept_jitter: tuple[float, float, float] = (0.025, 0.025, 0.015)
  target_xy: tuple[float, float] = (2.70, 0.0)
  target_jitter: tuple[float, float] = (0.15, 0.20)
  position_noise: float = 0.005
  velocity_noise: float = 0.03
  prediction_times: tuple[float, ...] = (0.1, 0.2, 0.3)
  max_ball_time: float = 3.0
  debug_table: bool = False
  difficulty_ramp_start_iteration: int = 0
  """Learning iteration at which optional hard-distribution ramp starts."""
  difficulty_ramp_steps: int = 0
  """Number of learning iterations to interpolate from baseline to configured ranges."""

  def build(self, env: ManagerBasedRlEnv) -> ReceivingBallCommand:
    return ReceivingBallCommand(self, env)


class ReceivingBallCommand(CommandTerm):
  """One incoming feed and at most one rewarded return per episode."""

  cfg: ReceivingBallCommandCfg
  pos_w: torch.Tensor
  vel_w: torch.Tensor
  intercept_w: torch.Tensor
  target_w: torch.Tensor
  landing_w: torch.Tensor
  _previous_racket: torch.Tensor
  _intercept_offset: torch.Tensor
  elapsed: torch.Tensor
  launch_time: torch.Tensor
  intercept_time: torch.Tensor
  min_racket_distance: torch.Tensor
  landing_error: torch.Tensor
  hit: torch.Tensor
  landed: torch.Tensor
  success: torch.Tensor
  missed: torch.Tensor
  net_cleared: torch.Tensor
  net_hit: torch.Tensor
  active: torch.Tensor
  hit_event: torch.Tensor
  net_event: torch.Tensor
  landing_event: torch.Tensor

  def __init__(self, cfg: ReceivingBallCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self._site_id = self.robot.find_sites((cfg.racket_site_name,), preserve_order=True)[
      0
    ][0]
    self._body_id = self.robot.body_names.index(cfg.racket_body_name)
    self._normal_local = torch.zeros(self.num_envs, 3, device=self.device)
    self._normal_local[:, cfg.racket_normal_axis] = 1
    self._offset = torch.tensor(cfg.racket_offset, device=self.device).expand(
      self.num_envs, 3
    )
    for name in (
      "pos_w",
      "vel_w",
      "intercept_w",
      "target_w",
      "landing_w",
      "_previous_racket",
      "_intercept_offset",
    ):
      setattr(self, name, torch.zeros(self.num_envs, 3, device=self.device))
    for name in (
      "elapsed",
      "launch_time",
      "intercept_time",
      "min_racket_distance",
      "landing_error",
    ):
      setattr(self, name, torch.zeros(self.num_envs, device=self.device))
    for name in (
      "hit",
      "landed",
      "success",
      "missed",
      "net_cleared",
      "net_hit",
      "active",
      "hit_event",
      "net_event",
      "landing_event",
    ):
      setattr(
        self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
      )
    self._reference_hit_pos: torch.Tensor | None = None
    self._reference_hit_time: torch.Tensor | None = None
    for name in (
      "hit_rate",
      "net_clear_rate",
      "return_success_rate",
      "miss_rate",
      "minimum_racket_distance",
      "landing_error",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def net_x(self) -> float:
    return 0.5 * (self.cfg.table_near_x + self.cfg.table_far_x)

  def _racket_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.robot.data.site_pos_w[:, self._site_id], quat_apply(
      self.robot.data.body_link_quat_w[:, self._body_id], self._normal_local
    )

  def _cache_reference_strikes(self) -> None:
    motion = cast(
      MotionCommand, self._env.command_manager.get_term(self.cfg.motion_command_name)
    )
    index = motion.cfg.body_names.index(self.cfg.racket_body_name)
    points, times = [], []
    # Reading the same files here only extracts small metadata; the motion loader
    # remains the source of transformed/interpolated body coordinates.
    paths = _resolve_motion_paths(motion.cfg)
    for clip, path in zip(motion.motion.motions, paths, strict=True):
      with np.load(path, allow_pickle=False) as data:
        if "strike_time_s" in data:
          t = float(np.asarray(data["strike_time_s"]).reshape(-1)[0])
        elif "strike_frame" in data:
          t = float(np.asarray(data["strike_frame"]).reshape(-1)[0]) / clip.fps
        elif "hope_strike_frame" in data:
          t = float(np.asarray(data["hope_strike_frame"]).reshape(-1)[0]) / clip.fps
        else:
          t = round((clip.time_step_total - 1) * self.cfg.strike_phase) / clip.fps
      frame = min(max(round(t * clip.fps), 0), clip.time_step_total - 1)
      q = clip.body_quat_w[frame, index].unsqueeze(0)
      p = clip.body_pos_w[frame, index] + quat_apply(q, self._offset[:1])[0]
      points.append(p)
      times.append(t)
    self._reference_hit_pos = torch.stack(points)
    self._reference_hit_time = torch.tensor(times, device=self.device)

  def _difficulty_scale(self) -> float:
    steps = int(self.cfg.difficulty_ramp_steps)
    if steps <= 0:
      return 1.0
    iteration = int(getattr(self._env, "current_learning_iteration", 0))
    start = int(self.cfg.difficulty_ramp_start_iteration)
    return float(max(0.0, min(1.0, (iteration - start) / steps)))

  @staticmethod
  def _lerp_tuple(base, target, scale: float) -> tuple[float, ...]:
    return tuple(float(b + scale * (t - b)) for b, t in zip(base, target, strict=True))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self._reference_hit_pos is None:
      self._cache_reference_strikes()
    motion = cast(
      MotionCommand, self._env.command_manager.get_term(self.cfg.motion_command_name)
    )
    ids = motion.motion_ids[env_ids]
    origins = self._env.scene.env_origins[env_ids]
    count = len(env_ids)
    assert self._reference_hit_pos is not None and self._reference_hit_time is not None
    difficulty = self._difficulty_scale()
    intercept_jitter = self._lerp_tuple(
      (0.025, 0.025, 0.015), self.cfg.intercept_jitter, difficulty
    )
    target_jitter = self._lerp_tuple((0.15, 0.20), self.cfg.target_jitter, difficulty)
    flight_time_range = self._lerp_tuple(
      (0.4, 0.5), self.cfg.flight_time_range, difficulty
    )
    speed_range = self._lerp_tuple(
      (3.5, 4.5), self.cfg.incoming_speed_range, difficulty
    )
    lateral_speed_range = self._lerp_tuple(
      (0.0, 0.0), self.cfg.incoming_lateral_speed_range, difficulty
    )
    arrival_time_jitter = difficulty * self.cfg.arrival_time_jitter
    minimum_intercept_time = 0.35 + difficulty * (self.cfg.minimum_intercept_time - 0.35)
    jitter = torch.tensor(intercept_jitter, device=self.device)
    intercept = (
      self._reference_hit_pos[ids]
      + origins
      + (2 * torch.rand(count, 3, device=self.device) - 1) * jitter
    )
    remaining = self._reference_hit_time[ids] - motion.current_time_seconds(env_ids)
    if arrival_time_jitter > 0:
      remaining += (2 * torch.rand_like(remaining) - 1) * arrival_time_jitter
    remaining = remaining.clamp_min(minimum_intercept_time)
    flight = torch.empty(count, device=self.device).uniform_(
      *flight_time_range
    )
    flight = torch.minimum(flight, remaining)
    speed = torch.empty(count, device=self.device).uniform_(
      *speed_range
    )
    final_velocity = torch.zeros(count, 3, device=self.device)
    final_velocity[:, 0] = -speed
    final_velocity[:, 1] = torch.empty(count, device=self.device).uniform_(
      *lateral_speed_range
    )
    final_velocity[:, 2] = self.cfg.incoming_vertical_speed
    velocity = final_velocity.clone()
    velocity[:, 2] += self.cfg.gravity * flight
    position = intercept - velocity * flight[:, None]
    position[:, 2] += 0.5 * self.cfg.gravity * flight**2
    if self.cfg.drag:
      # Shooting adjustment preserves the actual drag dynamics, rather than
      # snapping the ball to its target at the planned interception time.
      for _ in range(4):
        predicted_p, predicted_v = position.clone(), velocity.clone()
        for _ in range(24):
          predicted_p, predicted_v = ballistic_step(
            predicted_p, predicted_v, flight / 24, self.cfg.gravity, self.cfg.drag
          )
        velocity += (intercept - predicted_p) / flight[:, None]
    self.pos_w[env_ids] = position
    self.vel_w[env_ids] = velocity
    self.intercept_w[env_ids] = intercept
    self._intercept_offset[env_ids] = intercept - self._reference_hit_pos[ids] - origins
    self.intercept_time[env_ids] = remaining
    self.launch_time[env_ids] = remaining - flight
    self.elapsed[env_ids] = 0
    target = (
      torch.tensor(
        (*self.cfg.target_xy, self.cfg.table_height + self.cfg.ball_radius),
        device=self.device,
      )
      .expand(count, 3)
      .clone()
    )
    target[:, :2] += (2 * torch.rand(count, 2, device=self.device) - 1) * torch.tensor(
      target_jitter, device=self.device
    )
    self.target_w[env_ids] = target + origins
    self.landing_w[env_ids] = 0
    self.landing_error[env_ids] = 0
    self.min_racket_distance[env_ids] = 10
    for name in (
      "hit",
      "landed",
      "success",
      "missed",
      "net_cleared",
      "net_hit",
      "active",
      "hit_event",
      "net_event",
      "landing_event",
    ):
      getattr(self, name)[env_ids] = False
    # Reset occurs before sim.forward: use the freshly written reference pose,
    # avoiding an apparent first-substep paddle velocity from the previous episode.
    self._previous_racket[env_ids] = motion.body_pos_w[
      env_ids, motion.cfg.body_names.index(self.cfg.racket_body_name)
    ] + quat_apply(
      motion.body_quat_w[
        env_ids, motion.cfg.body_names.index(self.cfg.racket_body_name)
      ],
      self._offset[env_ids],
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # CommandManager resets before its regular metric update. Capture the final
    # substep's event latches here, so rates count episodes rather than dwell time.
    self._update_metrics()
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    self.metrics["hit_rate"].copy_(self.hit.float())
    self.metrics["net_clear_rate"].copy_(self.net_cleared.float())
    self.metrics["return_success_rate"].copy_(self.success.float())
    self.metrics["miss_rate"].copy_(self.missed.float())
    self.metrics["minimum_racket_distance"].copy_(self.min_racket_distance)
    self.metrics["landing_error"].copy_(self.landing_error)

  def _update_command(self) -> None:
    # ManagerBasedRlEnv evaluates rewards before command updates. Events thus
    # accumulate over all physics substeps and pay exactly once per policy step.
    self.hit_event.zero_()
    self.net_event.zero_()
    self.landing_event.zero_()

  def compute_substep(self, dt: float) -> None:
    center, normal = self._racket_pose()
    previous_elapsed = self.elapsed.clone()
    self.elapsed += dt
    self.active = (self.elapsed >= self.launch_time) & ~self.missed & ~self.landed
    # A delayed launcher releases inside a substep with the correct flight time.
    live_dt = (self.elapsed - torch.maximum(previous_elapsed, self.launch_time)).clamp(
      0, dt
    )
    origins = self._env.scene.env_origins
    p0 = self.pos_w - origins
    p1, v1 = ballistic_step(p0, self.vel_w, live_dt, self.cfg.gravity, self.cfg.drag)
    reflected_p, reflected_v, contact = swept_racket_collision(
      p0,
      p1,
      v1,
      self._previous_racket - origins,
      center - origins,
      normal,
      dt,
      ball_radius=self.cfg.ball_radius,
      racket_radius=self.cfg.racket_radius,
      restitution=self.cfg.racket_restitution,
    )
    contact &= self.active & ~self.hit
    p1 = torch.where(contact[:, None], reflected_p, p1)
    v1 = torch.where(contact[:, None], reflected_v, v1)
    self.hit |= contact
    self.hit_event |= contact
    p1, v1, net_hit, net_clear = swept_net_collision(
      p0,
      p1,
      v1,
      dt,
      net_x=self.net_x,
      width=self.cfg.table_width,
      table_height=self.cfg.table_height,
      net_height=self.cfg.net_height,
      ball_radius=self.cfg.ball_radius,
    )
    self.net_hit |= net_hit & self.active
    self.net_event |= net_clear & self.active & self.hit & ~self.net_cleared
    self.net_cleared |= net_clear & self.active & self.hit
    p1, v1, table_hit, landing = swept_table_collision(
      p0,
      p1,
      v1,
      dt,
      near_x=self.cfg.table_near_x,
      far_x=self.cfg.table_far_x,
      width=self.cfg.table_width,
      height=self.cfg.table_height,
      ball_radius=self.cfg.ball_radius,
      restitution=self.cfg.table_restitution,
    )
    outgoing_landing = table_hit & self.active & self.hit & ~self.landed
    self.landing_w[outgoing_landing] = (landing + origins)[outgoing_landing]
    self.landing_error[outgoing_landing] = (
      (landing + origins - self.target_w)[:, :2]
    ).norm(dim=-1)[outgoing_landing]
    self.success |= (
      outgoing_landing & self.net_cleared & ~self.net_hit & (landing[:, 0] > self.net_x)
    )
    self.landing_event |= outgoing_landing & self.success
    self.landed |= outgoing_landing
    out = (
      (p1[:, 2] < self.cfg.ball_radius)
      | (p1[:, 0] < -1.0)
      | (p1[:, 0] > self.cfg.table_far_x + 2.0)
      | (p1[:, 1].abs() > 2.5)
      | (self.elapsed > self.cfg.max_ball_time)
    )
    self.missed |= self.active & (out | net_hit | (outgoing_landing & ~self.success))
    self.pos_w = torch.where(self.active[:, None], p1 + origins, self.pos_w)
    self.vel_w = torch.where(self.active[:, None], v1, self.vel_w)
    distance = (self.pos_w - center).norm(dim=-1)
    self.min_racket_distance = torch.where(
      self.active,
      torch.minimum(self.min_racket_distance, distance),
      self.min_racket_distance,
    )
    self._previous_racket.copy_(center)

  @property
  def command(self) -> torch.Tensor:
    root_position = self.robot.data.root_link_pos_w
    root_quaternion = self.robot.data.root_link_quat_w
    position = self.pos_w
    velocity = self.vel_w
    if self.cfg.position_noise:
      position = (
        position + (2 * torch.rand_like(position) - 1) * self.cfg.position_noise
      )
    if self.cfg.velocity_noise:
      velocity = (
        velocity + (2 * torch.rand_like(velocity) - 1) * self.cfg.velocity_noise
      )
    components = [
      quat_apply_inverse(root_quaternion, position - root_position),
      quat_apply_inverse(root_quaternion, velocity),
      quat_apply_inverse(root_quaternion, self.intercept_w - root_position),
      (self.intercept_time - self.elapsed).clamp_min(0)[:, None],
      quat_apply_inverse(root_quaternion, self.target_w - root_position),
    ]
    for duration in self.cfg.prediction_times:
      predicted, _ = ballistic_step(
        position, velocity, duration, self.cfg.gravity, self.cfg.drag
      )
      components.append(quat_apply_inverse(root_quaternion, predicted - root_position))
    components.append(self.hit.float()[:, None])
    return torch.cat(components, dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for index in visualizer.get_env_indices(self.num_envs):
      origin = self._env.scene.env_origins[index].cpu().numpy()
      midpoint = 0.5 * (self.cfg.table_near_x + self.cfg.table_far_x)
      if self.cfg.debug_table:
        visualizer.add_box(
          origin + np.array([midpoint, 0, self.cfg.table_height - 0.015]),
          np.array(
            [
              (self.cfg.table_far_x - self.cfg.table_near_x) / 2,
              self.cfg.table_width / 2,
              0.015,
            ]
          ),
          np.eye(3),
          (0.12, 0.35, 0.6, 0.7),
        )
        visualizer.add_box(
          origin
          + np.array([self.net_x, 0, self.cfg.table_height + self.cfg.net_height / 2]),
          np.array([0.004, self.cfg.table_width / 2, self.cfg.net_height / 2]),
          np.eye(3),
          (0.7, 0.7, 0.7, 0.55),
        )
      visualizer.add_sphere(
        self.pos_w[index].cpu().numpy(), self.cfg.ball_radius, (1.0, 0.65, 0.1, 1.0)
      )
      visualizer.add_sphere(
        self.target_w[index].cpu().numpy(), 0.035, (0.2, 1.0, 0.3, 0.7)
      )


def _ball(env: ManagerBasedRlEnv, command_name: str) -> ReceivingBallCommand:
  return cast(ReceivingBallCommand, env.command_manager.get_term(command_name))


def ball_observation(
  env: ManagerBasedRlEnv, command_name: str = "ball"
) -> torch.Tensor:
  return _ball(env, command_name).command


def ball_approach_reward(
  env: ManagerBasedRlEnv, command_name: str = "ball", std: float = 0.25
) -> torch.Tensor:
  ball = _ball(env, command_name)
  center, _ = ball._racket_pose()
  distance2 = (ball.pos_w - center).square().sum(-1)
  return torch.exp(-distance2 / std**2) * ball.active * ~ball.hit


def ball_hit_reward(env: ManagerBasedRlEnv, command_name: str = "ball") -> torch.Tensor:
  return _ball(env, command_name).hit_event.float()


def ball_net_reward(env: ManagerBasedRlEnv, command_name: str = "ball") -> torch.Tensor:
  return _ball(env, command_name).net_event.float()


def ball_landing_reward(
  env: ManagerBasedRlEnv, command_name: str = "ball", std: float = 0.5
) -> torch.Tensor:
  ball = _ball(env, command_name)
  return ball.landing_event.float() * torch.exp(-ball.landing_error.square() / std**2)


def ball_intercept_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "ball",
  std: float = 0.15,
  time_window: float = 0.4,
) -> torch.Tensor:
  """Track the stroke on the physical ball schedule, independent of chosen phase.

  Using ``motion.body_pos_w`` here lets the planner slow its reference and collect
  easy tracking reward while the real ball passes. Instead, the scheduled stroke
  reaches its annotated strike at the independently sampled ball arrival time.
  """
  ball = _ball(env, command_name)
  motion = cast(
    MotionCommand, env.command_manager.get_term(ball.cfg.motion_command_name)
  )
  wrist_id = motion.cfg.body_names.index(ball.cfg.racket_body_name)
  assert ball._reference_hit_time is not None
  nominal_strike = ball._reference_hit_time[motion.motion_ids]
  scheduled_time = ball.elapsed / ball.intercept_time.clamp_min(1e-6) * nominal_strike
  scheduled_position = motion.motion._gather_interp(
    "body_pos_w", motion.motion_ids, scheduled_time
  )[:, wrist_id]
  scheduled_quaternion = motion.motion._gather_interp(
    "body_quat_w", motion.motion_ids, scheduled_time
  )[:, wrist_id]
  reference = (
    scheduled_position
    + env.scene.env_origins
    + quat_apply(scheduled_quaternion, ball._offset)
  )
  desired = reference + ball._intercept_offset
  center, _ = ball._racket_pose()
  spatial = torch.exp(-(center - desired).square().sum(-1) / std**2)
  timing = torch.exp(-((ball.elapsed - ball.intercept_time) / time_window).square())
  return spatial * timing * ~ball.hit * ~ball.missed


def ball_normal_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "ball",
  flight_time: float = 0.65,
  time_window: float = 0.25,
) -> torch.Tensor:
  """Shape face alignment toward a ballistic target-return impulse near impact.

  This is guidance only: success still requires the actual swept collision and
  actual opponent-table bounce. No velocity is modified by this reward.
  """
  ball = _ball(env, command_name)
  _, normal = ball._racket_pose()
  desired_outgoing = (ball.target_w - ball.intercept_w) / flight_time
  desired_outgoing[:, 2] += 0.5 * ball.cfg.gravity * flight_time
  remaining = (ball.intercept_time - ball.elapsed).clamp_min(0)
  incoming = ball.vel_w.clone()
  incoming[:, 2] -= ball.cfg.gravity * remaining
  impulse = desired_outgoing - incoming
  impulse = impulse / impulse.norm(dim=-1, keepdim=True).clamp_min(1e-8)
  alignment = (normal * impulse).sum(-1).abs().square()
  timing = torch.exp(-((ball.elapsed - ball.intercept_time) / time_window).square())
  return alignment * timing * ~ball.hit * ~ball.missed


def ball_missed(env: ManagerBasedRlEnv, command_name: str = "ball") -> torch.Tensor:
  return _ball(env, command_name).missed

"""Portable snapshot of the training ball physics (no mjlab runtime imports).

Origin: src/mjlab/tasks/a3_pingpong/ball.py. Contract tests compare both modules.
"""

import torch


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


def planner_table_flight(
  position: torch.Tensor,
  velocity: torch.Tensor,
  dt: float,
  *,
  gravity: float,
  drag: float,
  horizontal: float,
  vertical: float,
  near_x: float,
  far_x: float,
  width: float,
  height: float,
  ball_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """One flight step of the HOPE A30808 planner's constant-bounce ball model.

  Port of hope_planner_optimized_2 BallTrajectoryPredictor._predict (bounce_model
  "constant"): second-order flight under quadratic drag; a descending centre
  that reaches one radius above the table, inside its bounds widened by one
  radius, bounces at the interpolated instant with that instant's velocity,
  v+ = diag(C_h, C_h, -C_v) v-, and the rest of the step continues with the
  post-bounce acceleration. Unlike the planner's prediction, an off-table ball
  keeps falling. Returns position, velocity, contact, contact point and the
  contact's time into the step.
  """

  def acceleration(v):
    a = -drag * v.norm(dim=-1, keepdim=True) * v
    a = a.clone()
    a[..., 2] -= gravity
    return a

  a = acceleration(velocity)
  position1 = position + velocity * dt + 0.5 * a * dt**2
  velocity1 = velocity + a * dt
  plane = height + ball_radius
  drop = position[:, 2] - position1[:, 2]
  fraction = torch.where(
    drop > 1e-9, (position[:, 2] - plane) / drop.clamp_min(1e-9), 0.5
  ).clamp(0, 1)
  point = position + fraction[:, None] * (position1 - position)
  point[:, 2] = plane
  contact = (
    (position1[:, 2] < plane)
    & (velocity1[:, 2] < 0)
    & (point[:, 0] >= near_x - ball_radius)
    & (point[:, 0] <= far_x + ball_radius)
    & (point[:, 1].abs() <= width / 2 + ball_radius)
  )
  at_contact = velocity + a * (fraction * dt)[:, None]
  scale = torch.tensor([horizontal, horizontal, -vertical], dtype=velocity.dtype)
  bounced = at_contact * scale
  rest = ((1 - fraction) * dt)[:, None]
  a_bounced = acceleration(bounced)
  return (
    torch.where(
      contact[:, None], point + bounced * rest + 0.5 * a_bounced * rest**2, position1
    ),
    torch.where(contact[:, None], bounced + a_bounced * rest, velocity1),
    contact,
    point,
    fraction * dt,
  )


# ``ball.flight_model`` value selecting planner_table_flight over the training model.
PLANNER_FLIGHT = "hope_planner_constant"


def planner_flight_options(cfg):
  """planner_table_flight keywords for a ball config with a Y-centred table."""
  return dict(
    gravity=cfg["gravity"],
    drag=cfg["drag"],
    horizontal=cfg["table_tangential_damping"],
    vertical=cfg["table_restitution"],
    near_x=cfg["table_near_x"],
    far_x=cfg["table_far_x"],
    width=cfg["table_width"],
    height=cfg["table_height"],
    ball_radius=cfg["ball_radius"],
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

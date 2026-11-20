"""Interchangeable raw-command motion selectors.

W is hope_table world (positions in metres); H is the current pelvis yaw-only
heading frame (X forward, Y left). Neither selector modifies robot state.
"""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.motions import RebasedMotion, heading
from deploy.motion_20261119.reference.hope_g1.a3_planner import ReceivePlan
from deploy.motion_20261119.reference.planner_interface import PlannerInterfaceError, validate_selector_interface
from deploy.motion_20261119.reference.racket_command import RacketCommand, RacketCommandArray


def check_command_interface(selector, command):
  expected = RacketCommandArray if selector.mode == "distance_command_array" else RacketCommand
  if not isinstance(command, expected):
    raise PlannerInterfaceError(f"selector={selector.mode} requires {expected.__name__}, got {type(command).__name__}")


def selection_timing(executor, command, now, *, live_update):
  """Live refinements use the existing 40 ms guard, not the initial entry window."""
  remaining = command.deadline - now
  indices = np.asarray(executor.candidate_indices)
  starts = executor.clips.start_frames(indices, remaining)
  maximum = executor.a3_options.get("max_command_time_to_strike", .5)
  feasible = (
    (.04 < remaining <= maximum) & (starts >= 0)
    if live_update else (0 < remaining <= maximum) & executor.clips.startable(indices, starts)
  )
  return remaining, indices, starts, feasible


def execution_reference(executor, state, index, *, live_update):
  # An unchanged motion must retain its world transform, even as the robot moves.
  if live_update and index == executor.motion_index:
    return executor.reference
  if executor.reference_mode == "a3_live":
    if index == executor.home_index:
      return executor.home
    return RebasedMotion(executor.library.motions[index], state.anchor_pos, state.anchor_quat)
  return (executor.references or executor.aligned_library(state))[index]


class DistanceSelector:
  """Distance/start-frame/hysteresis ranking for initial and live selection."""

  mode = "distance"

  def __init__(self, config):
    pass

  def choose(self, executor, state, command, now, *, live_update=False):
    """Select using only a public command, independent of its producer."""
    check_command_interface(self, command)
    remaining, indices, starts, feasible = selection_timing(
      executor, command, now, live_update=live_update,
    )
    strikes = executor.candidate_strikes(state, indices)
    if live_update:
      # Score precisely the reference that would execute: frozen home/current,
      # and current-state rebases for other motions. Keep the bulk query vectorized.
      for index, reference in ((executor.home_index, executor.home),
                               (executor.motion_index, executor.reference)):
        strikes[indices == index] = reference.strike_position(executor.cfg["ball"])
    distances = np.linalg.norm(
      command.position - strikes, axis=1
    )
    diagnostic = dict(
      command.diagnostic,
      remaining=remaining,
      target_mode=executor.target_mode,
      feasible_clips=int(feasible.sum()),
    )
    if not feasible.any():
      nearest = int(np.argmin(distances))
      executor.decision = dict(
        diagnostic,
        reason="arrival_outside_start_window",
        start_frame=int(starts[nearest]),
        motion_index=int(indices[nearest]),
      )
      return None
    feasible = np.flatnonzero(feasible)
    # Clips aiming at the same point differ only in swing duration: prefer the
    # one entered closest to its first frame, which equals the idle home pose.
    tie = executor.options.get("start_frame_tie_distance", 0.01)
    group = feasible[distances[feasible] <= distances[feasible].min() + tie]
    best = min(group, key=lambda k: (starts[k], distances[k], indices[k]))
    current = feasible[indices[feasible] == executor.motion_index]
    if (
      current.size
      and best != current[0]
      and not distances[best] + executor.options.get("selection_hysteresis", 0.05)
      < distances[current[0]]
    ):
      best = current[0]
    distance, index = float(distances[best]), int(indices[best])
    reference = execution_reference(executor, state, index, live_update=live_update)
    if distance > executor.options["max_intercept_distance"]:
      executor.decision = dict(
        reason="a3_intercept_unreachable",
        distance=distance,
        motion_index=index,
        command=command.diagnostic,
      )
      return None
    diagnostic.update(start_frame=int(starts[best]), motion_index=index)
    executor.decision = dict(diagnostic, reason="accepted")
    return distance, index, reference, ReceivePlan(command, diagnostic)

  def check_update(self, executor, state, command, now):
    check_command_interface(self, command)
    distance = float(np.linalg.norm(command.position - executor.reference.strike_position(executor.cfg["ball"])))
    return dict(reason="live_command_unreachable" if distance > executor.options["max_intercept_distance"]
                else "accepted", distance=distance)


class DistanceCommandArraySelector(DistanceSelector):
  """Distance ranking over every feasible (command slot, motion) pair."""

  mode = "distance_command_array"

  def choose(self, executor, state, command, now, *, live_update=False, fixed_motion=False):
    check_command_interface(self, command)
    valid = []
    for slot, cmd in enumerate(command.commands):
      if cmd is None or not cmd.diagnostic.get("valid", True):
        continue
      vectors = [np.asarray(v) for v in (cmd.position, cmd.velocity, cmd.normal)]
      if (any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors)
          or np.linalg.norm(vectors[2]) < 1e-6 or not np.isfinite(cmd.deadline)):
        continue
      valid.append((slot, cmd))
    common = dict(command.diagnostic, selector_mode=self.mode,
                  command_count=len(command.commands), valid_commands=len(valid),
                  center_index=command.center_index)
    if not valid:
      executor.decision = dict(common, reason="command_array_no_valid_commands")
      return None
    indices = np.asarray([executor.motion_index] if fixed_motion else executor.candidate_indices)
    remaining = np.array([cmd.deadline-now for _, cmd in valid])
    starts = executor.clips.start_frames(indices[None, :], remaining[:, None])
    maximum = executor.a3_options.get("max_command_time_to_strike", .5)
    feasible = ((remaining[:, None] > (.04 if live_update else 0))
                & (remaining[:, None] <= maximum))
    feasible = feasible & ((starts >= 0) if live_update else executor.clips.startable(indices, starts))
    strikes = executor.candidate_strikes(state, indices)
    if live_update:
      for index, reference in ((executor.home_index, executor.home),
                               (executor.motion_index, executor.reference)):
        strikes[indices == index] = reference.strike_position(executor.cfg["ball"])
    positions = np.array([cmd.position for _, cmd in valid])
    distances = np.linalg.norm(positions[:, None, :] - strikes[None, :, :], axis=2)
    common["feasible_pairs"] = int(feasible.sum())
    if not feasible.any():
      executor.decision = dict(common, reason="arrival_outside_start_window")
      return None
    rows, cols = np.nonzero(feasible)
    tie = executor.options.get("start_frame_tie_distance", .01)
    pair_distances = distances[rows, cols]
    slots = np.array([slot for slot, _ in valid])

    def rank(mask):
      # Keep ranking vectorized even with thousands of clips per command.
      r, c = rows[mask], cols[mask]
      order = np.lexsort((slots[r], indices[c], distances[r, c], starts[r, c]))
      return int(r[order[0]]), int(c[order[0]])

    best = rank(pair_distances <= pair_distances.min()+tie)
    # Compare with the best command for the current motion, using the same
    # start-frame tie rule before applying the original motion hysteresis.
    current = indices[cols] == executor.motion_index
    if current.any():
      current_min = pair_distances[current].min()
      incumbent = rank(current & (pair_distances <= current_min+tie))
      if (indices[best[1]] != executor.motion_index
          and not distances[best]+executor.options.get("selection_hysteresis", .05) < distances[incumbent]):
        best = incumbent
    row, col = best
    slot, selected = valid[row]
    index, distance = int(indices[col]), float(distances[best])
    diagnostic = dict(selected.diagnostic)
    diagnostic.update(common, command_index=slot, motion_index=index,
                      remaining=float(remaining[row]), start_frame=int(starts[best]), distance=distance)
    if distance > executor.options["max_intercept_distance"]:
      executor.decision = dict(diagnostic, reason="a3_intercept_unreachable")
      return None
    reference = execution_reference(executor, state, index, live_update=live_update)
    executor.decision = dict(diagnostic, reason="accepted")
    return distance, index, reference, ReceivePlan(selected, dict(executor.decision))


class HeightBandSelector:
  """Narrow world-Z strips extended in XY by bounded base translations.

  Movement budgets are screening heuristics, not a calibrated dynamics model
  or a proof of balance. Exact commanded positions are preserved.
  """

  mode = "height_band"

  def __init__(self, config):
    self.half_width = float(config.get("height_half_width_m", .04))
    self.height_tie = float(config.get("height_tie_tolerance_m", .01))
    self.max_shift = np.asarray(config.get("max_base_shift_heading_m", [.15, .45]), dtype=float)
    self.max_speed = np.asarray(config.get("base_speed_heading_mps", [.35, .70]), dtype=float)
    self.settle = float(config.get("settle_seconds", .08))
    self.switch_penalty = float(config.get("switch_penalty", .05))
    if (self.max_shift.shape != (2,) or self.max_speed.shape != (2,)
        or not np.isfinite(np.r_[self.half_width, self.height_tie, self.max_shift,
                                self.max_speed, self.settle, self.switch_penalty]).all()
        or self.half_width <= 0 or not 0 <= self.height_tie <= self.half_width
        or np.any(self.max_shift <= 0) or np.any(self.max_speed <= 0)
        or self.settle < 0 or self.switch_penalty < 0):
      raise ValueError("Invalid height_band selector widths/movement budgets")

  def evaluate(self, executor, state, command, now, reference, index):
    nominal = executor.reference_goal(reference, index)
    base_w = nominal.base_xy + (command.position - nominal.position)[:2]
    delta_w = base_w - state.root_pos[:2]
    yaw = heading(state.root_quat)
    c, s = np.cos(yaw), np.sin(yaw)
    # W -> H: rotate a displacement, not an absolute world position.
    delta_h = np.array([[c, s], [-s, c]]) @ delta_w
    height_error = float(command.position[2] - nominal.position[2])
    available = max(command.deadline - now - self.settle, 0.)
    budget = np.minimum(self.max_shift, self.max_speed * available)
    diagnostic = dict(
      motion_index=int(index), reference_strike_world=nominal.position.tolist(),
      height_error_m=height_error,
      height_band_world_z=[float(nominal.position[2]-self.half_width), float(nominal.position[2]+self.half_width)],
      base_target_world_xy=base_w.tolist(), base_delta_world_xy=delta_w.tolist(),
      base_delta_heading_xy=delta_h.tolist(), base_budget_heading_xy=budget.tolist(),
      distance=float(np.linalg.norm(command.position-nominal.position)),
    )
    if abs(height_error) > self.half_width + 1e-9:
      reason = "height_outside_band"
    elif np.any(np.abs(delta_h) > self.max_shift + 1e-9):
      reason = "base_shift_exceeded"
    elif np.any(np.abs(delta_h) > budget + 1e-9):
      reason = "base_time_budget_exceeded"
    else:
      reason = "accepted"
    diagnostic.update(reason=reason, movement_cost=float(np.sum((delta_h/np.maximum(budget, 1e-9))**2)))
    return diagnostic

  def choose(self, executor, state, command, now, *, live_update=False):
    check_command_interface(self, command)
    remaining, indices, starts, feasible = selection_timing(
      executor, command, now, live_update=live_update,
    )
    candidates, accepted = [], []
    for index, frame, valid_time in zip(indices, starts, feasible, strict=True):
      index, frame = int(index), int(frame)
      if not valid_time:
        candidates.append(dict(motion_index=index, start_frame=frame, reason="arrival_outside_start_window"))
        continue
      reference = execution_reference(executor, state, index, live_update=live_update)
      diagnostic = self.evaluate(executor, state, command, now, reference, index)
      diagnostic["start_frame"] = frame
      diagnostic["selection_cost"] = diagnostic["movement_cost"] + (
        self.switch_penalty if index != executor.motion_index else 0.)
      candidates.append(diagnostic)
      if diagnostic["reason"] == "accepted":
        accepted.append((reference, diagnostic))
    common = dict(selector_mode=self.mode, remaining=remaining,
                  feasible_clips=int(feasible.sum()), candidates=candidates)
    if not accepted:
      executor.decision = dict(common, reason="height_band_no_feasible_motion")
      return None
    best_height = min(abs(d["height_error_m"]) for _, d in accepted)
    group = [(r,d) for r,d in accepted if abs(d["height_error_m"]) <= best_height+self.height_tie+1e-9]
    reference, diagnostic = min(group, key=lambda item: (
      item[1]["selection_cost"], abs(item[1]["height_error_m"]), item[1]["start_frame"], item[1]["motion_index"]))
    executor.decision = dict(command.diagnostic)
    executor.decision.update(common)
    executor.decision.update(diagnostic)
    return (diagnostic["distance"], diagnostic["motion_index"], reference,
            ReceivePlan(command, dict(executor.decision)))

  def check_update(self, executor, state, command, now):
    check_command_interface(self, command)
    diagnostic = self.evaluate(executor, state, command, now, executor.reference, executor.motion_index)
    diagnostic["selector_mode"] = self.mode
    return diagnostic


SELECTORS = {cls.mode: cls for cls in (DistanceSelector, HeightBandSelector, DistanceCommandArraySelector)}


def build_motion_selector(options):
  config = options.get("motion_selector", {})
  if not isinstance(config, dict):
    raise ValueError("motion_selector must be a mapping")
  mode = config.get("mode", "distance")
  if not isinstance(mode, str) or mode not in SELECTORS:
    raise ValueError(f"motion_selector.mode must be one of {list(SELECTORS)}")
  validate_selector_interface(options)
  settings = config.get(mode, {})
  if not isinstance(settings, dict):
    raise ValueError(f"motion_selector.{mode} must be a mapping")
  return SELECTORS[mode](settings)


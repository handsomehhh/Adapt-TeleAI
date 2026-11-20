"""World-frame task command shared by planners, selectors and benchmarks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RacketCommand:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  deadline: float
  diagnostic: dict


@dataclass
class RacketCommandArray:
  """One published batch, preserving invalid slots as None and ROS indices.

  Indices address the received array, not the planner's pre-filter grid.
  ``center_index == -1`` means the original hitting plane was filtered out.
  """

  commands: tuple[RacketCommand | None, ...]
  center_index: int
  diagnostic: dict

  def __post_init__(self):
    if (not self.commands or isinstance(self.center_index, bool)
        or not isinstance(self.center_index, int)
        or not -1 <= self.center_index < len(self.commands)):
      raise ValueError("Invalid RacketCommandArray slots/center_index")

  @property
  def deadline(self):
    return max((cmd.deadline for cmd in self.commands if cmd is not None), default=-np.inf)

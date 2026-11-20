"""Explicit simulation IO boundary for the shared hardware control loop."""

from dataclasses import dataclass, field
from typing import Any, Callable


class SimulationReset(Exception):
  """Operator requested a new simulation attempt, never a robot command."""


@dataclass
class SimulationIO:
  clock: Any
  robot_factory: Callable
  mocap_factory: Callable
  on_controller: Callable
  on_record: Callable
  metadata: dict = field(default_factory=dict)
  observe_physical_limits: bool = False

"""Planner wire contracts, shared by the ROS bridge and deployment process."""

SINGLE_COMMAND = "hope_msgs/msg/RacketCommand"
COMMAND_ARRAY = "hope_msgs/msg/RacketCommandArray"
COMMAND_TYPES = (SINGLE_COMMAND, COMMAND_ARRAY)
ARRAY_SELECTOR = "distance_command_array"


class PlannerInterfaceError(RuntimeError):
  """A planner and its consumer disagree; this must not become a dropped packet."""


def command_type(config):
  default = COMMAND_ARRAY if config.get("target_source") == "hope_planner_command_array" else SINGLE_COMMAND
  result = config.get("planner_command_type", default)
  if result not in COMMAND_TYPES:
    raise ValueError(f"planner_command_type must be one of {COMMAND_TYPES}")
  return result


def validate_selector_interface(options):
  interface = command_type(options)
  selector = options.get("motion_selector", {})
  if not isinstance(selector, dict):
    raise ValueError("motion_selector must be a mapping")
  mode = selector.get("mode", "distance")
  if (interface == COMMAND_ARRAY) != (mode == ARRAY_SELECTOR):
    raise PlannerInterfaceError(
      f"{COMMAND_ARRAY} and motion_selector.mode={ARRAY_SELECTOR} must be used together; "
      f"got interface={interface}, selector={mode}"
    )
  if interface == COMMAND_ARRAY and (
    options.get("target_source", "reference") == "reference"
    or options.get("planner_transport", "embedded") != "ros2"
    or options.get("a3", {}).get("target_mode", "raw") != "raw"
  ):
    raise PlannerInterfaceError("distance_command_array requires raw commands from an external ROS planner")
  return interface

"""Check SDK imports and HG message serialization without opening DDS channels."""

import importlib
from pathlib import Path


def sdk_status():
  try:
    importlib.import_module("unitree_sdk2py.core.channel")
    defaults = importlib.import_module("unitree_sdk2py.idl.default")
    messages = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    crc_module = importlib.import_module("unitree_sdk2py.utils.crc")
    command = defaults.unitree_hg_msg_dds__LowCmd_()
    command.crc = crc_module.CRC().Crc(command)
    payload = command.serialize()
    decoded = messages.LowCmd_.deserialize(payload)
    if decoded.crc != command.crc or len(decoded.motor_cmd) < 29:
      raise ValueError("G1 HG command serialization check failed")
    return {"available": True, "error": None, "hg_command_bytes": len(payload)}
  except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
    return {"available": False, "error": str(exc)}


def require_sdk():
  status = sdk_status()
  if not status["available"]:
    raise RuntimeError(
      "Unitree SDK2 cannot be imported in this Python environment: "
      f"{status['error']}. Install it with: uv pip install --python .venv/bin/python "
      "-e deploy/third_party/unitree_sdk2_python; see deploy/README.md."
    )


def network_interfaces():
  directory = Path("/sys/class/net")
  return sorted(path.name for path in directory.iterdir()) if directory.exists() else []

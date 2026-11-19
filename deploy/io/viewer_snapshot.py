"""Small atomic snapshots written by recording workers, never the control loop."""

import json
import os
from pathlib import Path

from deploy.io.diagnostics import json_default


def boot_id():
  return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def write_snapshot(path, payload):
  path = Path(path)
  temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
  try:
    temporary.write_text(json.dumps(payload, allow_nan=False, default=json_default))
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)

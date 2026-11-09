"""Locate shared planner code relative to the deployment checkout in all modes."""

import importlib
import sys
from pathlib import Path


def use_sibling_planner():
  """Prefer sibling source over installed copies, independently of cwd."""
  deployment_root = Path(__file__).resolve().parents[2]
  source = deployment_root.parent / "g1_planner_ros2/src/hope_planner_g1"
  package = source / "hope_planner_g1"
  if not (package / "__init__.py").is_file():
    raise FileNotFoundError(
      f"G1 planner source not found: {package}. "
      "Place g1_planner_ros2 beside the deployment checkout (Adapt-Deploy or Adapt-TeleAI)."
    )
  loaded = sys.modules.get("hope_planner_g1")
  if loaded is not None and Path(loaded.__file__).resolve().parent != package.resolve():
    raise ImportError(
      f"Another G1 planner is already loaded from {loaded.__file__}; "
      f"restart this process to use {package}."
    )
  if str(source) in sys.path:
    sys.path.remove(str(source))
  sys.path.insert(0, str(source))
  importlib.invalidate_caches()
  importlib.import_module("hope_planner_g1")
  return source

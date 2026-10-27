"""Entry point that runs the A30808 ROS node copied into deploy."""

import os
from pathlib import Path
import sys


def main(args=None):
  configured_root = os.environ.get("DEPLOY_G1_PINGPONG_ROOT")
  if configured_root:
    repo_root = Path(configured_root).resolve()
  else:
    repo_root = next(
      (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "deploy" / "hope_g1" / "hope_planner_optimized_2" / "node.py").is_file()
      ),
      None,
    )
    if repo_root is None:
      raise RuntimeError("Set DEPLOY_G1_PINGPONG_ROOT to the deploy repository root")
  sys.path.insert(0, str(repo_root))

  from deploy.hope_g1.hope_planner_optimized_2.node import main as a30808_main

  a30808_main(args)

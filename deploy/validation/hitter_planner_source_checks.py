"""Relocatable embedded-planner loading, without an installed planner package."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from deploy.hope_g1_hitter import planner_source


def run_relocated(tmp_path, *, sibling, checkout="Adapt-TeleAI"):
  # Different checkout location and cwd, no site-packages (-S), plus a stale
  # package on sys.path: only the explicit sibling source should be imported.
  helper = tmp_path / "relocated" / checkout / "deploy/hope_g1_hitter/planner_source.py"
  helper.parent.mkdir(parents=True)
  shutil.copyfile(planner_source.__file__, helper)
  package = tmp_path / "relocated/g1_planner_ros2/src/hope_planner_g1/hope_planner_g1"
  if sibling:
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("origin = 'sibling'\n")
  stale = tmp_path / "old_install/hope_planner_g1"
  stale.mkdir(parents=True)
  (stale / "__init__.py").write_text("raise RuntimeError('stale planner imported')\n")
  code = """
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
spec = importlib.util.spec_from_file_location('planner_source', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
source = helper.use_sibling_planner()
import hope_planner_g1
assert hope_planner_g1.origin == 'sibling'
assert Path(hope_planner_g1.__file__).parent == source / 'hope_planner_g1'
assert helper.use_sibling_planner() == source
print(source)
"""
  return subprocess.run(
    [sys.executable, "-S", "-c", code, str(helper), str(stale.parent)],
    cwd=tmp_path,
    capture_output=True,
    text=True,
  )


@pytest.mark.parametrize("checkout", ["Adapt-TeleAI", "Adapt-Deploy"])
def test_relocated_checkout_loads_sibling_without_installation(tmp_path, checkout):
  result = run_relocated(tmp_path, sibling=True, checkout=checkout)
  assert result.returncode == 0, result.stderr
  assert str(tmp_path / "relocated/g1_planner_ros2") in result.stdout


def test_missing_sibling_does_not_load_stale_installation(tmp_path):
  result = run_relocated(tmp_path, sibling=False)
  assert result.returncode != 0
  assert "Place g1_planner_ros2 beside" in result.stderr
  assert "stale planner imported" not in result.stderr


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_physics_resolution_bootstraps_planner_in_fresh_process(tmp_path, mode):
  # A stale install must never mask a missing bootstrap in real/check mode.
  stale = tmp_path / "hope_planner_g1"
  stale.mkdir()
  (stale / "__init__.py").write_text("raise RuntimeError('stale planner imported')\n")
  code = """
import sys
from pathlib import Path
import yaml
from deploy.hope_g1_hitter.configuration import resolve_physics
from deploy.hope_g1_hitter import planner_source
sys.path.insert(0, sys.argv[1])
directory = Path(planner_source.__file__).parent
filename = 'rally_mujoco.yaml' if sys.argv[2] == 'sim' else 'rally.yaml'
options = yaml.safe_load((directory / 'configs' / filename).read_text())
resolved = resolve_physics(options, sys.argv[2])
import hope_planner_g1
assert Path(hope_planner_g1.__file__).parent == Path(resolved['planner_source']) / 'hope_planner_g1'
assert resolved['runtime_mode'] == sys.argv[2]
assert resolved['planner_transport'] == options['planner_transport']
print(resolved['planner_source'])
"""
  result = subprocess.run(
    [sys.executable, "-c", code, str(tmp_path), mode],
    cwd=Path(__file__).resolve().parents[2],
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
  assert "g1_planner_ros2/src/hope_planner_g1" in result.stdout

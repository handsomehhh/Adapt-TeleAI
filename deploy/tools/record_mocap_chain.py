"""Launch read-only ROS capture in system Python, isolated from policy Python."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "deploy/hope_g1_hitter/configs/chain_recording.yaml"


class ChainRecording:
  def __init__(self, output, config=DEFAULT_CONFIG, mocap_config=None):
    output = Path(output)
    if output.suffix == ".jsonl":
      output = output.parent / (output.stem + "_diagnostics")
    self.directory = output / "ros_chain"
    self.directory.mkdir(parents=True, exist_ok=False)
    self.process = self.log = None
    self.closed = False
    cfg = yaml.safe_load(Path(config).read_text())
    if mocap_config:
      mocap = yaml.safe_load(Path(mocap_config).read_text())
      cfg["ball_topic"] = mocap.get("source", {}).get("ball_topic", cfg["ball_topic"])
      cfg["command_topic"] = mocap.get("planner_command_topic", cfg["command_topic"])
      cfg["feedback_topic"] = mocap.get("planner_feedback", {}).get("topic", cfg["feedback_topic"])
    effective = self.directory / "config.yaml"
    effective.write_text(yaml.safe_dump(cfg))
    planner = ROOT.parent / "g1_planner_ros2"
    setup = planner / "install/setup.bash"
    source = planner / "src/hope_planner_g1"
    # Only the child gets this environment. Preserve DDS settings from the caller.
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PATH"] = "/usr/bin:/bin"
    env.setdefault("ROS_DOMAIN_ID", str(cfg.get("ros_domain_id", 23)))
    env.setdefault("ROS_LOCALHOST_ONLY", "0")
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    command = [
      "/bin/bash", "--noprofile", "--norc", "-c",
      'source "$1" && source "$2" && exec /usr/bin/python3 -c '
      '\'import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); '
      'runpy.run_module("hope_planner_g1.chain_recorder",run_name="__main__")\' '
      '"$3" --output "$4" --config "$5"',
      "chain-recorder", str(Path(cfg["ros_setup"]).expanduser()), str(setup),
      str(source), str(self.directory.resolve()), str(effective.resolve()),
    ]
    self.log = (self.directory / "process.log").open("w")
    self.process = subprocess.Popen(command, env=env, stdout=self.log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 10
    while not (self.directory / "ready.json").exists():
      if self.process.poll() is not None or time.monotonic() > deadline:
        self.close()
        raise RuntimeError(f"ROS chain recorder failed; see {self.directory / 'process.log'}")
      time.sleep(0.05)
    meta = json.loads((self.directory / "metadata.json").read_text())
    print(f"ROS chain log: {self.directory}; VRPN tap={meta['vrpn_tap']}", flush=True)

  def close(self):
    if self.closed:
      return
    self.closed = True
    forced = False
    if self.process is not None and self.process.poll() is None:
      self.process.terminate()
      try:
        self.process.wait(timeout=8)
      except subprocess.TimeoutExpired:
        forced = True
        # This is our private session, containing only the recorder and its tap.
        os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=2)
    if self.log is not None:
      self.log.close()
    code = self.process.returncode if self.process is not None else None
    (self.directory / "launcher_summary.json").write_text(json.dumps(
      dict(exitcode=code, forced_stop=forced), indent=2,
    ))
    if code != 0 or forced:
      print(f"ROS chain recording did not finish cleanly: {self.directory / 'process.log'}",
            flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--duration", type=float, default=30)
  args = parser.parse_args()
  if args.duration <= 0:
    parser.error("duration must be positive")
  recorder = ChainRecording(args.output, args.config)
  try:
    time.sleep(args.duration)
  except KeyboardInterrupt:
    pass
  finally:
    recorder.close()


if __name__ == "__main__":
  main()

"""Bounded real HTTP/Viser startup checks; both server processes are reaped."""

import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from deploy import ROOT


def check(mode, output, port, trace=None):
  command = [
    sys.executable,
    "-u",
    "-m",
    "deploy",
    "--mode",
    mode,
    "--port",
    str(port),
    "--output",
    str(output),
  ]
  if trace:
    command.extend(["--trace", str(trace)])
  log_path = output.with_suffix(".log")
  with log_path.open("x") as log:
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
      for _ in range(200):
        if process.poll() is not None:
          raise RuntimeError(log_path.read_text())
        if "Open http://" in log_path.read_text():
          with urllib.request.urlopen(
            f"http://127.0.0.1:{port}", timeout=5
          ) as response:
            assert response.status == 200 and len(response.read()) > 100
          return {
            "mode": mode,
            "http": 200,
            "controls_built": True,
            "log": str(log_path),
          }
        time.sleep(0.1)
      raise TimeoutError(log_path.read_text())
    finally:
      if process.poll() is None:
        process.send_signal(signal.SIGINT)
      try:
        process.wait(timeout=15)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
  root = ROOT / "deploy/recordings" / time.strftime("web_validation_%Y%m%d_%H%M%S")
  root.mkdir(parents=True)
  # Generate a short genuine recording before checking its replay viewer.
  subprocess.run(
    [
      sys.executable,
      "-m",
      "deploy",
      "--mode",
      "tracker",
      "--viewer",
      "headless",
      "--steps",
      "8",
      "--output",
      str(root / "recording"),
    ],
    check=True,
    cwd=ROOT,
  )
  results = [
    check("tracker", root / "tracker", 18087),
    check("hierarchy", root / "hierarchy", 18085),
    check("replay", root / "replay", 18086, root / "recording/trace.npz"),
  ]
  (root / "web_checks.json").write_text(json.dumps(results, indent=2) + "\n")
  print(json.dumps({"results": results, "output": str(root)}, indent=2))


if __name__ == "__main__":
  main()

"""Launcher boundaries, process lifecycle and HTTP control integration."""

import json
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
from deploy.tools.web_platform import Platform, build_command, make_server


def test_modes_keep_existing_entry_points_and_hide_future_trajectory(tmp_path):
  recording = tmp_path / "deploy/recordings/hope_g1/real_test"
  recording.mkdir(parents=True)
  (recording / "trace.jsonl").write_text("")
  cmd, output = build_command(
    dict(mode="replay", recording=str(recording), start=137.9, rate=0.5),
    8071,
    "127.0.0.1",
    root=tmp_path,
  )
  assert "deploy.tools.view_real_log" in cmd
  assert "--paused" in cmd and cmd[cmd.index("--start") + 1] == "137.9"
  assert output is None
  command, output = build_command(
    dict(mode="command", y=-0.8, z=0.3), 8071, "127.0.0.1", root=tmp_path
  )
  assert "--command-only" in command and "--point" in command
  assert output.is_relative_to(tmp_path / "deploy/recordings/web_runs")
  receive, _ = build_command(
    dict(mode="receive", limit=3), 8071, "127.0.0.1", root=tmp_path
  )
  assert "deploy.benchmarks.receive.run" in receive
  assert not {"--point", "--command-only", "--recording"}.intersection(receive)
  sim, output = build_command(dict(mode="simulation"), 8071, "127.0.0.1", root=tmp_path)
  assert sim[sim.index("--env") + 1] == "mujoco"
  assert "manual" in sim and "real" not in sim and output is None


@pytest.mark.parametrize(
  "fields",
  [
    dict(mode="real"),
    dict(mode="command", y="nan"),
    dict(mode="receive", limit=1.5),
    dict(mode="receive", limit=True),
    dict(mode="receive", rate=0),
    dict(mode="command", adapter="a:b;echo oops"),
    dict(mode="receive", suite="/etc/passwd"),
  ],
)
def test_invalid_parameters_cannot_launch(fields, tmp_path):
  with pytest.raises(ValueError):
    build_command(fields, 8071, "127.0.0.1", root=tmp_path)


def test_http_controls_process_cleanup_reports_and_path_boundaries(
  tmp_path, monkeypatch
):
  from deploy.tools import web_platform

  output = tmp_path / "deploy/recordings/web_runs/test"
  program = (
    "import signal,time; "
    "signal.signal(signal.SIGINT,lambda *_:exit(0)); "
    "print('worker ready',flush=True); time.sleep(60)"
  )
  monkeypatch.setattr(
    web_platform,
    "build_command",
    lambda *a, **k: ([sys.executable, "-u", "-c", program], output),
  )
  platform = Platform(0, "127.0.0.1", root=tmp_path)
  server = make_server(platform, "127.0.0.1", 0)
  worker = threading.Thread(target=server.serve_forever, daemon=True)
  worker.start()
  base = f"http://127.0.0.1:{server.server_port}"
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

  def get(path):
    return opener.open(base + path, timeout=5)

  def post(path, token=None):
    return opener.open(
      urllib.request.Request(
        base + path,
        data=b'{"mode":"receive"}',
        headers={"X-Platform-Token": token or platform.token},
      ),
      timeout=5,
    )

  try:
    assert platform.token.encode() in get("/").read()
    with pytest.raises(urllib.error.HTTPError) as rejected:
      post("/api/start", "invalid")
    assert rejected.value.code == 403 and platform.process is None
    assert json.loads(post("/api/start").read())["status"] == "running"
    first = platform.process
    with pytest.raises(urllib.error.HTTPError) as duplicate:
      post("/api/start")
    assert duplicate.value.code == 400 and platform.process is first
    deadline = time.monotonic() + 5
    while (
      "worker ready" not in platform.snapshot()["logs"] and time.monotonic() < deadline
    ):
      time.sleep(0.02)
    assert "worker ready" in platform.snapshot()["logs"]
    post("/api/stop")
    first.wait(timeout=5)
    assert first.returncode == 0
    assert platform.snapshot()["status"] == "stopped"
    (output / "report.html").write_text("<h1>report</h1>")
    report = platform.snapshot()["report"]
    assert b"report" in get(report).read()
    assert json.loads(get("/api/catalog").read())["reports"] == [
      str((output / "report.html").relative_to(tmp_path))
    ]
    for path in ("/artifacts/../../etc/passwd", "/artifacts/%2Fetc%2Fpasswd"):
      with pytest.raises(urllib.error.HTTPError) as missing:
        get(path)
      assert missing.value.code == 404
    # Stopping an old job must never signal a newly launched job.
    post("/api/start")
    assert platform.process is not first
    assert platform.snapshot()["generation"] == 2
  finally:
    platform.stop()
    if platform.process is not None:
      try:
        platform.process.wait(timeout=5)
      except Exception:
        platform._signal(platform.process, signal.SIGKILL)
    server.shutdown()
    server.server_close()
    worker.join(timeout=5)

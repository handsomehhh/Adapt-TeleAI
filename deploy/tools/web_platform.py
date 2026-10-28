"""Local browser launcher for the existing simulation and replay entry points."""

import argparse
import hmac
import json
import math
import mimetypes
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("web_platform.html")
MODES = {
  "command": "deploy.benchmarks.racket_command.run",
  "receive": "deploy.benchmarks.receive.run",
  "replay": "deploy.tools.view_real_log",
  "simulation": "deploy.hope_g1.deploy_agent",
}


def number(data, key, default, low, high, integer=False):
  value = data.get(key, default)
  if isinstance(value, bool):
    raise ValueError(f"{key} 必须是数值")
  try:
    value = float(value)
  except (TypeError, ValueError):
    raise ValueError(f"{key} 必须是数值") from None
  if not math.isfinite(value) or not low <= value <= high:
    raise ValueError(f"{key} 必须在 {low} 到 {high} 之间")
  if integer and value != int(value):
    raise ValueError(f"{key} 必须是整数")
  return int(value) if integer else value


def local_path(value, *, root=ROOT, directory=False, recordings_only=False):
  if not isinstance(value, str) or not value:
    raise ValueError("请选择有效路径")
  path = (root / value).resolve()
  boundary = root / "deploy/recordings" if recordings_only else root
  if not path.is_relative_to(boundary.resolve()):
    raise ValueError("路径必须位于仓库允许的目录内")
  if not (path.is_dir() if directory else path.is_file()):
    raise ValueError(f"路径不存在：{value}")
  return path


def build_command(data, viewer_port, host, *, root=ROOT):
  """Validate form fields and use argv directly; never execute a shell string."""
  mode = data.get("mode")
  if mode not in MODES:
    raise ValueError("未知测试模式")
  command = [sys.executable, "-u", "-m", MODES[mode]]
  command += ["--viewer", "web", "--web-host", host, "--web-port", str(viewer_port)]
  rate = number(data, "rate", 1, 0.0625, 8)
  output = None
  if mode == "replay":
    path = local_path(
      data.get("recording"), root=root, directory=True, recordings_only=True
    )
    if not (path / "trace.jsonl").is_file():
      raise ValueError("录制目录缺少 trace.jsonl")
    command += [
      "--recording",
      str(path),
      "--start",
      str(number(data, "start", 0, 0, 1e7)),
    ]
    command += ["--rate", str(rate), "--paused"]
  elif mode == "simulation":
    command += ["--env", "mujoco", "--render", "--sim-controls", "manual"]
    command += ["--duration", str(number(data, "duration", 600, 1, 86400))]
  else:
    output = (
      root
      / "deploy/recordings/web_runs"
      / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + mode)
    )
    command += ["--render", "--output", str(output), "--playback-rate", str(rate)]
    command += ["--limit", str(number(data, "limit", 1, 1, 100000, integer=True))]
    suite = data.get("suite", "")
    if suite:
      command += ["--suite", str(local_path(suite, root=root))]
    else:
      command += ["--seed", str(number(data, "seed", 42, 0, 2**32 - 1, integer=True))]
      if mode == "command":
        command += [
          "--point",
          str(number(data, "y", -0.8, -10, 10)),
          str(number(data, "z", 0.3, -10, 10)),
        ]
        command += ["--directions", "1", "--speed-levels", "1", "--tts-levels", "1"]
    if mode == "command":
      command += ["--command-only"]
      adapter = data.get("adapter", "")
      if adapter:
        if (
          not isinstance(adapter, str)
          or ":" not in adapter
          or any(
            not part.isidentifier() for part in adapter.replace(":", ".").split(".")
          )
        ):
          raise ValueError("adapter 应为 Python 模块:类名")
        command += ["--adapter", adapter]
    for key in ("task", "rally", "runtime", "mocap"):
      if data.get(key):
        command += [f"--{key}-config", str(local_path(data[key], root=root))]
  return command, output


class Platform:
  def __init__(self, viewer_port, host, *, root=ROOT):
    self.root, self.viewer_port, self.host = root, viewer_port, host
    self.token = secrets.token_urlsafe(32)
    self.lock = threading.RLock()
    self.process = None
    self.command = []
    self.output = None
    self.mode = None
    self.logs = deque(maxlen=1200)
    self.stopping = False
    self.generation = 0

  def start(self, data):
    with self.lock:
      if self.process is not None and self.process.poll() is None:
        raise ValueError("已有测试运行，请先停止当前测试")
      command, output = build_command(data, self.viewer_port, self.host, root=self.root)
      # Viser otherwise silently chooses another port, breaking the embedded view.
      with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
          probe.bind((self.host, self.viewer_port))
        except OSError:
          raise ValueError(
            f"三维端口 {self.viewer_port} 已占用，请更换 --viewer-port"
          ) from None
      env = {**os.environ, "PYTHONUNBUFFERED": "1"}
      process = subprocess.Popen(
        command,
        cwd=self.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
      )
      self.process, self.command, self.output = process, command, output
      self.mode, self.stopping = data["mode"], False
      self.logs.clear()
      self.generation += 1
      threading.Thread(target=self._read, args=(process, output), daemon=True).start()

  def _read(self, process, output):
    # Keep full launch evidence on disk, while bounding the live browser log.
    stream = None
    try:
      if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        stream = (output / "launcher.log").open("w")
      for line in process.stdout:
        if stream is not None:
          stream.write(line)
          stream.flush()
        with self.lock:
          if self.process is process:
            self.logs.append(line.rstrip())
    finally:
      if stream is not None:
        stream.close()
      process.stdout.close()
      process.wait()

  def stop(self):
    with self.lock:
      process = self.process
      if process is None or process.poll() is not None or self.stopping:
        return
      self.stopping = True
      self._signal(process, signal.SIGINT)
      threading.Thread(target=self._finish_stop, args=(process,), daemon=True).start()

  @staticmethod
  def _signal(process, sig):
    try:
      os.killpg(process.pid, sig)
    except ProcessLookupError:
      pass

  def _finish_stop(self, process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
      try:
        process.wait(timeout=8)
        return
      except subprocess.TimeoutExpired:
        self._signal(process, sig)
    process.wait()

  def snapshot(self):
    with self.lock:
      code = self.process.poll() if self.process else None
      running = self.process is not None and code is None
      status = (
        "stopping"
        if running and self.stopping
        else "running"
        if running
        else (
          "idle"
          if self.process is None
          else "stopped"
          if self.stopping
          else "complete"
          if code == 0
          else "failed"
        )
      )
      output = str(self.output.relative_to(self.root)) if self.output else None
      report = None
      if self.output and (self.output / "report.html").is_file():
        report = "/artifacts/" + output + "/report.html"
      return dict(
        status=status,
        exit_code=code,
        mode=self.mode,
        command=self.command,
        output=output,
        report=report,
        logs=list(self.logs),
        viewer_port=self.viewer_port,
        generation=self.generation,
      )


def make_server(platform, host, port):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
      pass

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
      if not isinstance(body, bytes):
        body = json.dumps(body, ensure_ascii=False).encode()
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.send_header("X-Content-Type-Options", "nosniff")
      self.end_headers()
      try:
        self.wfile.write(body)
      except (BrokenPipeError, ConnectionResetError):
        pass

    def do_GET(self):
      path = unquote(urlsplit(self.path).path)
      if path == "/":
        body = PAGE.read_text().replace("__TOKEN__", platform.token).encode()
        self.respond(200, body, "text/html; charset=utf-8")
      elif path == "/api/status":
        self.respond(200, platform.snapshot())
      elif path == "/api/catalog":
        base = platform.root / "deploy/recordings"
        recordings = sorted(
          str(p.parent.relative_to(platform.root))
          for p in (base / "hope_g1").glob("*/trace.jsonl")
        )
        suites = sorted(
          str(p.relative_to(platform.root)) for p in base.glob("*/*/suite.json")
        )
        reports = sorted(
          str(p.relative_to(platform.root)) for p in base.glob("*/*/report.html")
        )
        self.respond(200, dict(recordings=recordings, suites=suites, reports=reports))
      elif path.startswith("/artifacts/"):
        try:
          file = local_path(
            path.removeprefix("/artifacts/"), root=platform.root, recordings_only=True
          )
          if file.suffix.lower() not in {
            ".html",
            ".json",
            ".jsonl",
            ".csv",
            ".png",
            ".svg",
            ".log",
          }:
            raise ValueError("不支持的文件类型")
          self.respond(
            200, file.read_bytes(), mimetypes.guess_type(file)[0] or "text/plain"
          )
        except ValueError as error:
          self.respond(404, dict(error=str(error)))
      else:
        self.respond(404, dict(error="Not found"))

    def do_POST(self):
      if not hmac.compare_digest(
        self.headers.get("X-Platform-Token", ""), platform.token
      ):
        self.respond(403, dict(error="请刷新平台页面后重试"))
        return
      try:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 16384:
          raise ValueError("请求大小不合法")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
          raise ValueError("请求必须为对象")
        if self.path == "/api/start":
          platform.start(data)
        elif self.path == "/api/stop":
          platform.stop()
        else:
          self.respond(404, dict(error="Not found"))
          return
        self.respond(200, platform.snapshot())
      except (ValueError, TypeError, OSError) as error:
        self.respond(400, dict(error=str(error)))

  return ThreadingHTTPServer((host, port), Handler)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8070)
  parser.add_argument("--viewer-port", type=int, default=8071)
  args = parser.parse_args()
  if args.port == args.viewer_port or not all(
    1 <= p <= 65535 for p in (args.port, args.viewer_port)
  ):
    parser.error("平台和 viewer 需使用不同的有效端口")
  platform = Platform(args.viewer_port, args.host)
  server = make_server(platform, args.host, args.port)
  print(f"乒乓球测试平台：http://{args.host}:{server.server_port}", flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    platform.stop()
    if platform.process is not None:
      try:
        platform.process.wait(timeout=18)
      except subprocess.TimeoutExpired:
        platform._signal(platform.process, signal.SIGKILL)
    server.server_close()


if __name__ == "__main__":
  main()

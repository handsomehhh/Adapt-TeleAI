"""Boundary diagnosis regressions; no ROS graph or robot required."""

import json
import math
from types import SimpleNamespace

from deploy.tools.monitor_planner import Recorder, classify, encode


def state(**overrides):
  values = dict(uptime=30, planner_present=True, pose_age=.01, raw_age=.01,
                diag_age=.1, progress_age=.1, command_age=20,
                valid_progress_age=20, inbound_points=0, monitor_gap=1)
  values.update(overrides)
  return classify(**values)[0]


def test_idle_without_commands_is_not_a_fault():
  assert state() == "observing_no_commands"
  assert state(command_age=math.inf) == "observing_no_commands"


def test_distinguishes_upstream_relay_executor_and_subscription():
  assert state(pose_age=4, raw_age=4) == "ball_input_gap"
  assert state(pose_age=4) == "relay_input_gap"
  assert state(diag_age=4) == "planner_heartbeat_gap"
  assert state(progress_age=4) == "planner_receive_stalled"


def test_output_suspicion_requires_new_planner_counter_progress():
  assert state(valid_progress_age=.1) == "command_delivery_gap"
  assert state(command_age=.1, valid_progress_age=.1) == "commands_observed"


def test_inbound_without_commands_is_only_a_suspicion():
  assert state(inbound_points=20) == "inbound_without_commands"
  assert state(inbound_points=19) == "observing_no_commands"


def test_startup_shutdown_and_monitor_delay_do_not_claim_planner_deadlock():
  assert state(uptime=1, pose_age=math.inf) == "warming_up"
  assert state(planner_present=False, diag_age=math.inf) == "waiting_for_planner"
  assert state(monitor_gap=5, diag_age=5) == "monitor_delayed"


def test_json_encodes_missing_samples_and_nan_as_null():
  assert json.loads(encode({"age": math.inf, "tts": math.nan})) == {"age": None, "tts": None}


def test_counter_baseline_and_restart_do_not_create_false_publication(tmp_path):
  recorder = Recorder(tmp_path, tmp_path / "absent.yaml", None)
  def diagnostic(received, valid):
    return SimpleNamespace(status=[SimpleNamespace(
      name="hope_planner_command_array", level=b"\x00", message="running",
      values=[SimpleNamespace(key="poses_received", value=str(received)),
              SimpleNamespace(key="valid_commands", value=str(valid))])])
  recorder.diagnostics(diagnostic(1000, 500))
  assert recorder.valid_progress_time == -math.inf
  recorder.diagnostics(diagnostic(1001, 501))
  assert recorder.valid_progress_time > 0
  recorder.diagnostics(diagnostic(1, 0))
  assert recorder.valid_progress_time == -math.inf


def test_only_plausible_inbound_points_count_and_duplicate_stamps_are_recorded(tmp_path):
  recorder = Recorder(tmp_path, tmp_path / "absent.yaml", None)
  recorder.position("poses", 1., [1., -.7, .5])
  recorder.position("poses", 1.01, [.95, -.7, .5])
  assert len(recorder.inbound) == 1
  recorder.position("poses", 1.01, [.90, -.7, .5])
  recorder.position("poses", 1.02, [.90, -.7, .5])
  recorder.position("poses", 1.03, [.85, .7, .5])
  assert len(recorder.inbound) == 1
  assert recorder.timestamp_anomalies["poses_duplicate"] == 1

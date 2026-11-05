#!/usr/bin/env bash
# Start a read-only monitor in a user service, independent of this terminal.
# Status: journalctl --user -u planner-readonly-monitor.service -f
# Stop:   systemctl --user stop planner-readonly-monitor.service
# Logs:   ~/.local/state/planner-monitor/latest/
# This is a transient service: rerun this script after logout/reboot.
set -euo pipefail

unit=planner-readonly-monitor.service
if systemctl --user is-active --quiet "$unit"; then
  echo "Already running: $unit"
  echo "Logs: $HOME/.local/state/planner-monitor/latest/"
  exit 0
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
monitor_base="$HOME/.local/state/planner-monitor"
monitor_output="$monitor_base/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -p "$monitor_output"

# Match the current deployment domain. The previous LOCALHOST_ONLY=23 is
# interpreted by this ROS installation as disabled, so use its explicit value 0.
systemd-run --user --collect --unit="$unit" \
  --property=Restart=on-failure --property=RestartSec=5 \
  --setenv=ROS_DOMAIN_ID=23 --setenv=ROS_LOCALHOST_ONLY=0 \
  --setenv=RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  /bin/bash -c 'source /opt/ros/humble/setup.bash && source /home/lxz/HOPE/hope_ws/install/setup.bash && exec /usr/bin/python3 "$1" --output "$2"' \
  planner-monitor "$repo_root/deploy/tools/monitor_planner.py" "$monitor_output"
ln -sfn "$monitor_output" "$monitor_base/latest"
echo "Logs: $monitor_output"

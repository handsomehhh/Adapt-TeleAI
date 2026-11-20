#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$deploy_root"
deploy_unit="adapt-g1-deploy-viewer"
deploy_action="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
case "$deploy_action" in
  start)
    mkdir -p deploy/recordings/service
    deploy_output="deploy/recordings/web_$(date +%Y%m%d_%H%M%S)"
    exec systemd-run --user --unit="$deploy_unit" --collect \
      --description="G1 AdaPT CPU deployment viewer" \
      --property=Type=exec \
      --property="WorkingDirectory=$deploy_root" \
      --property="StandardOutput=append:$deploy_root/deploy/recordings/service/viewer.log" \
      --property=StandardError=inherit \
      --property=KillSignal=SIGINT \
      --property=TimeoutStopSec=30 \
      --setenv="ADAPT_UV=$(command -v uv)" \
      --setenv="ADAPT_PYTHON=${ADAPT_PYTHON:-}" \
      /bin/bash "$deploy_root/deploy/run.sh" \
      --config deploy/configs/round1.json --output "$deploy_output" "$@"
    ;;
  stop) exec systemctl --user stop "$deploy_unit.service" ;;
  status) exec systemctl --user status "$deploy_unit.service" --no-pager ;;
  *) echo "Usage: bash deploy/service.sh {start [options...]|stop|status}" >&2; exit 2 ;;
esac

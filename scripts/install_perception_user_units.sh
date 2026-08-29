#!/usr/bin/env bash
# Install and enable only the perception-only persistent user units.  This
# command never starts them, so it cannot disturb a running perception stack.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT}/config/systemd/user"
USER_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
TARGET_DIR="${USER_CONFIG_HOME}/systemd/user"
TARGET="taskplanner-perception-stack.target"
VIEWER="taskplanner-perception-quad-viewer.service"
ENABLE_LINGER=true

if [[ "${1:-}" == "--no-enable-linger" ]]; then
  ENABLE_LINGER=false
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--no-enable-linger]" >&2
  exit 2
fi

units=(
  taskplanner-perception-tool-trt-server.service
  taskplanner-perception-ingress.service
  taskplanner-perception-cam4-ingress.service
  taskplanner-perception-cam4-palm-pose.service
  taskplanner-perception-tool-cam3-ingress.service
  taskplanner-perception-hand-cam1-ingress.service
  taskplanner-perception-hand-cam3-ingress.service
  taskplanner-perception-blood-flir-ingress.service
  taskplanner-perception-right-ee-ingress.service
  taskplanner-perception-head-ingress.service
  taskplanner-perception-tool-head-ingress.service
  taskplanner-perception-left-ee-ingress.service
  taskplanner-perception-hand-right-ee-ingress.service
  taskplanner-perception-hand-head-ingress.service
  taskplanner-perception-hand-left-ee-ingress.service
  taskplanner-perception-hand-fusion.service
  taskplanner-perception-final-overlay.service
  taskplanner-perception-operator-quad.service
  taskplanner-perception-quad-viewer.service
  taskplanner-perception-stack.target
)
for unit in "${units[@]}"; do
  test -f "${SOURCE_DIR}/${unit}"
  install -D -m 0644 "${SOURCE_DIR}/${unit}" "${TARGET_DIR}/${unit}"
done

systemctl --user daemon-reload
systemctl --user enable "${TARGET}"
systemctl --user enable "${VIEWER}"
if [[ "${ENABLE_LINGER}" == true ]]; then
  if ! loginctl enable-linger "$(id -un)"; then
    sudo loginctl enable-linger "$(id -un)"
  fi
  if [[ "$(loginctl show-user "$(id -un)" -p Linger --value)" != "yes" ]]; then
    echo "Linger did not become enabled; refusing a non-persistent installation." >&2
    exit 1
  fi
fi

echo "Installed and enabled ${TARGET} and ${VIEWER}; no service was started."
echo "Boot-time user-unit persistence verified: $(loginctl show-user $(id -un) -p Linger)"

#!/usr/bin/env bash
# Explicit opt-in installation of the persistent host socket-buffer cap.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${ROOT}/config/sysctl.d/90-taskplanner-perception-dds.conf"
TARGET="/etc/sysctl.d/90-taskplanner-perception-dds.conf"

if [[ ! -f "${SOURCE}" ]]; then
  echo "missing ${SOURCE}" >&2
  exit 2
fi
sudo install -D -m 0644 "${SOURCE}" "${TARGET}"
sudo sysctl --system
echo "installed ${TARGET}; verify with: sysctl net.core.rmem_max net.core.wmem_max"

#!/usr/bin/env bash
# Start only Hand detection. Delegates to the portable root script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${ROOT}/scripts/run_hand_cam4.sh" "$@"

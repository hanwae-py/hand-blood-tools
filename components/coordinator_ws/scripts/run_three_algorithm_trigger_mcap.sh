#!/usr/bin/env bash
# MCAP + Tool v1.4 + Hand + real Blood. Delegates to the portable root script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${ROOT}/scripts/run_three_algorithm_trigger_mcap.sh" "$@"

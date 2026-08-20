#!/usr/bin/env bash
# Start only Tool v1.4. Delegates to the portable root script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${ROOT}/scripts/run_tool_v14.sh" "$@"

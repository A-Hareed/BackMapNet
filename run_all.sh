#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[Deprecated] run_all.sh has been renamed to BackMapNet.sh" >&2
exec "${SCRIPT_DIR}/BackMapNet.sh" "$@"

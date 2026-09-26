#!/usr/bin/env bash
set -euo pipefail
# No global Docker cleanup. This script never activates a different environment.
if [[ "${CONDA_DEFAULT_ENV:-}" != "wgm" ]]; then
  echo "Activate the Python environment first: conda activate wgm" >&2
  exit 1
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-preflight}" == "full-rsi" ]]; then
  shift
  exec python hgm.py "$@"
fi
if [[ $# -eq 0 ]]; then
  set -- preflight
fi
exec python pipeline.py "$@"

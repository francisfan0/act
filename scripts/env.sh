#!/bin/sh
# Shared environment for actsim runs. Source this; do not execute it.
if [ -n "${ROOT:-}" ]; then
  : "${ROOT}"
else
  ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
fi
ACT_HOME="${ACT_HOME:-$HOME/async}"
export ACT_HOME
export PATH="$ACT_HOME/bin:$PATH"
export QWEN_RUNTIME="$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing $PY; create the project virtualenv first" >&2
  exit 1
fi
export QWEN_PYTHON_HOME="$("$PY" -c 'import sys; print(sys.base_prefix)')"
export QWEN_PYTHON_PATH="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

if [ -z "${QWEN_BRIDGE_LIB:-}" ]; then
  QWEN_BRIDGE_LIB="$("$ROOT/scripts/build_bridge.sh")"
  export QWEN_BRIDGE_LIB
fi

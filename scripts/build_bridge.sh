#!/bin/sh
# Build the actsim external bridge as a shared library.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
ACT_HOME="${ACT_HOME:-$HOME/async}"
PY_CONFIG="${PY_CONFIG:-/opt/homebrew/opt/python@3.12/bin/python3.12-config}"

if [ ! -f "$ACT_HOME/include/act/actsim_ext.h" ]; then
  echo "actsim_ext.h not found under $ACT_HOME; install the ACT toolchain first" >&2
  exit 1
fi
if [ ! -x "$PY_CONFIG" ]; then
  echo "python3.12-config not found at $PY_CONFIG" >&2
  exit 1
fi

mkdir -p "$ROOT/build"
OUT="$ROOT/build/libqwen_bridge.dylib"

# shellcheck disable=SC2046
clang++ -std=c++17 -fPIC -dynamiclib \
  -I"$ACT_HOME/include" \
  $("$PY_CONFIG" --includes) \
  -o "$OUT" \
  "$ROOT/runtime/bridge/bridge.cc" \
  $("$PY_CONFIG" --ldflags --embed)

# actsim dlopens this library. Absolute dependencies keep dyld from needing
# the bridge's build directory on its search path.
if otool -L "$OUT" | awk 'NR>1 {print $1}' | grep -q '@rpath/'; then
  python_lib="$(otool -L "$OUT" | awk '/libpython/ {print $1; exit}')"
  resolved="$("$PY_CONFIG" --prefix)/lib/libpython3.12.dylib"
  if [ ! -f "$resolved" ]; then
    resolved="$(find "$("$PY_CONFIG" --prefix)" -name 'libpython3.12.dylib' | head -1)"
  fi
  if [ -n "${python_lib:-}" ] && [ -n "${resolved:-}" ] && [ -f "$resolved" ]; then
    install_name_tool -change "$python_lib" "$resolved" "$OUT"
  fi
fi

echo "$OUT"

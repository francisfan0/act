#!/bin/sh
# Run the decomposed ACT graph on real tensors.
#   scripts/validate_graph.sh layer <L> [case]            one block, fed the reference input of layer L
#   scripts/validate_graph.sh model [case] [steps]        prefill plus cached decode, checked against the reference
#   scripts/validate_graph.sh generate "<prompt>" [steps] any prompt, no reference (output not validated)
# Writes runs/<run_id>/{report.json,journal.jsonl,artifacts.jsonl,actsim.log,graph.act}.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/env.sh"

MODE="${1:-}"
unset QWEN_PROMPT QWEN_DECODE_STEPS
case "$MODE" in
  layer)
    LAYER="${2:?usage: $0 layer <L> [case]}"
    CASE="${3:-capital}"
    NAME="layer${LAYER}-${CASE}"
    ;;
  model)
    CASE="${2:-capital}"
    if [ -n "${3:-}" ]; then export QWEN_DECODE_STEPS="$3"; fi
    NAME="model-${CASE}"
    ;;
  generate)
    export QWEN_PROMPT="${2:?usage: $0 generate \"<prompt>\" [steps]}"
    export QWEN_DECODE_STEPS="${3:-16}"
    CASE=custom
    NAME="generate"
    ;;
  *)
    echo "usage: $0 layer <L> [case] | model [case] [steps] | generate \"<prompt>\" [steps]" >&2
    exit 2
    ;;
esac

if [ "$CASE" != custom ] && [ ! -f "$ROOT/runs/reference/$CASE/capture.pt" ]; then
  "$ROOT/.venv/bin/python" "$ROOT/reference/capture.py" "$CASE"
fi

QWEN_RUN_ID="$NAME-$(date +%Y%m%d-%H%M%S)"
QWEN_RUN_DIR="$ROOT/runs/$QWEN_RUN_ID"
QWEN_CASE="$CASE"
export QWEN_RUN_ID QWEN_RUN_DIR QWEN_CASE
mkdir -p "$QWEN_RUN_DIR"

ACT_FILE="$QWEN_RUN_DIR/graph.act"
if [ "$MODE" = layer ]; then
  "$PY" "$ROOT/scripts/gen_graph.py" layer "$LAYER" "$ACT_FILE"
else
  "$PY" "$ROOT/scripts/gen_graph.py" model "$ACT_FILE"
fi

LOG="$QWEN_RUN_DIR/actsim.log"
actsim -Wlang_subst:off -cnf="$ROOT/graph/qwen_extern.conf" \
  "$ACT_FILE" top < "$ROOT/graph/run.scr" > "$LOG" 2>&1 || {
  echo "actsim failed; log follows" >&2
  cat "$LOG" >&2
  exit 1
}

"$PY" "$ROOT/scripts/summarize_report.py" "$QWEN_RUN_DIR" "$LOG"

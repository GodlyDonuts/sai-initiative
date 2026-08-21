#!/bin/bash
#SBATCH --job-name=sai-screen-dev-mc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc36,evc38,evc43,evc50
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${GEOMETRY_SHA256:?GEOMETRY_SHA256 is required}"
: "${FAMILY:?FAMILY is required}"
: "${SHORT_SCREEN_RESULT:?SHORT_SCREEN_RESULT is required}"
: "${SHORT_SCREEN_RESULT_SHA256:?SHORT_SCREEN_RESULT_SHA256 is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${CHECKPOINT_MANIFEST:?CHECKPOINT_MANIFEST is required}"
: "${TRAINING_STREAM:?TRAINING_STREAM is required}"
: "${TRAINING_STREAM_IDENTITY:?TRAINING_STREAM_IDENTITY is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${TOKENIZER_SHA256:?TOKENIZER_SHA256 is required}"
: "${BENCHMARK:?BENCHMARK is required}"
: "${BENCHMARK_SOURCE:?BENCHMARK_SOURCE is required}"
: "${BENCHMARK_SOURCE_SHA256:?BENCHMARK_SOURCE_SHA256 is required}"
: "${DISJOINT_RECEIPT:?DISJOINT_RECEIPT is required}"
: "${DISJOINT_RECEIPT_SHA256:?DISJOINT_RECEIPT_SHA256 is required}"
: "${TRAINING_SOURCE_SHA256:?TRAINING_SOURCE_SHA256 is required}"
: "${EXPECTED_ROWS:?EXPECTED_ROWS is required}"
: "${EXPECTED_IDENTITY_ORDER_SHA256:?EXPECTED_IDENTITY_ORDER_SHA256 is required}"
: "${MAX_SEQUENCE_TOKENS:?MAX_SEQUENCE_TOKENS is required}"
: "${RUNTIME_PATH:?RUNTIME_PATH is required}"
: "${OUTPUT:?OUTPUT is required}"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in \
  "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" "$LOG_ROOT" \
  "$GEOMETRY" "$SHORT_SCREEN_RESULT" "$CHECKPOINT" "$CHECKPOINT_MANIFEST" \
  "$TRAINING_STREAM" "$TOKENIZER_ROOT" "$BENCHMARK_SOURCE" \
  "$DISJOINT_RECEIPT" "$RUNTIME_PATH" "$OUTPUT"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test -d "$TRAINING_STREAM"
test ! -L "$TRAINING_STREAM"
test -d "$TOKENIZER_ROOT"
test ! -L "$TOKENIZER_ROOT"
for path in \
  "$ENVIRONMENT_RECEIPT" "$GEOMETRY" "$SHORT_SCREEN_RESULT" "$CHECKPOINT" \
  "$CHECKPOINT_MANIFEST" "$BENCHMARK_SOURCE" "$DISJOINT_RECEIPT" \
  "$RUNTIME_PATH"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

"$PYTHON" -P -s -B - "$ENVIRONMENT_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["schema"] == "sai-fla-parity-receipt-v1"
assert receipt["status"] == "cuda_fla_parity_qualified"
assert receipt["parity_qualified"] is True
assert receipt["production_cuda_qualified"] is True
assert receipt["environment"]["device_type"] == "cuda"
assert receipt["environment"]["cuda_bf16_supported"] is True
assert receipt["operators"]["version"] == "0.4.2"
assert receipt["operators"]["mock"] is False
PY

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')"
test "$gpu_name" = "NVIDIA H100 PCIe"

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -P -s -B -m sai.evaluation.short_screen_mc \
  --geometry "$GEOMETRY" \
  --geometry-sha256 "$GEOMETRY_SHA256" \
  --family "$FAMILY" \
  --short-screen-result "$SHORT_SCREEN_RESULT" \
  --short-screen-result-sha256 "$SHORT_SCREEN_RESULT_SHA256" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --training-stream "$TRAINING_STREAM" \
  --training-stream-identity "$TRAINING_STREAM_IDENTITY" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --benchmark "$BENCHMARK" \
  --benchmark-source "$BENCHMARK_SOURCE" \
  --benchmark-source-sha256 "$BENCHMARK_SOURCE_SHA256" \
  --disjoint-receipt "$DISJOINT_RECEIPT" \
  --disjoint-receipt-sha256 "$DISJOINT_RECEIPT_SHA256" \
  --training-source-sha256 "$TRAINING_SOURCE_SHA256" \
  --expected-rows "$EXPECTED_ROWS" \
  --expected-identity-order-sha256 "$EXPECTED_IDENTITY_ORDER_SHA256" \
  --max-sequence-tokens "$MAX_SEQUENCE_TOKENS" \
  --runtime-path "$RUNTIME_PATH" \
  --runtime-path "$ENVIRONMENT_RECEIPT" \
  --runtime-path "$SAI_ROOT/src/sai/evaluation/short_screen_mc.py" \
  --output "$OUTPUT"

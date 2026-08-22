#!/bin/bash
#SBATCH --job-name=sai-learnability-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc35,evc36,evc38,evc43,evc50
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

required=(
  SAI_ROOT EXPECTED_COMMIT PYTHON FLA_ROOT LOG_ROOT GEOMETRY GEOMETRY_SHA256
  FAMILY PROBE_RESULT PROBE_RESULT_SHA256 STRONG_CHECKPOINT
  STRONG_CHECKPOINT_MANIFEST WEAK_MILESTONE WEAK_MILESTONE_STEP
  PROBE_TRAINING_STREAM PROBE_TRAINING_STREAM_IDENTITY TARGET_STREAM
  TARGET_STREAM_IDENTITY RUNTIME_RECEIPT RUNTIME_RECEIPT_SHA256 OUTPUT
)
for name in "${required[@]}"; do
  test -n "${!name:-}" || { echo "$name is required" >&2; exit 2; }
done

SCALE="${SCALE:-100m}"
BATCH_SIZE="${BATCH_SIZE:-1}"
case "$SCALE" in 100m|300m|1b) ;; *) exit 2 ;; esac
case "$BATCH_SIZE" in ''|*[!0-9]*|0) exit 2 ;; esac
case "$WEAK_MILESTONE_STEP" in ''|*[!0-9]*|0) exit 2 ;; esac

require_absolute() {
  case "$1" in /*) ;; *) echo "path must be absolute: $1" >&2; exit 2 ;; esac
}
for path in \
  "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$LOG_ROOT" "$GEOMETRY" \
  "$PROBE_RESULT" "$STRONG_CHECKPOINT" "$STRONG_CHECKPOINT_MANIFEST" \
  "$WEAK_MILESTONE" "$PROBE_TRAINING_STREAM" "$TARGET_STREAM" \
  "$RUNTIME_RECEIPT" "$OUTPUT"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"
for path in "$PROBE_TRAINING_STREAM" "$TARGET_STREAM"; do
  test -d "$path" && test ! -L "$path"
done
for path in \
  "$GEOMETRY" "$PROBE_RESULT" "$STRONG_CHECKPOINT" \
  "$STRONG_CHECKPOINT_MANIFEST" "$WEAK_MILESTONE" "$RUNTIME_RECEIPT"; do
  test -f "$path" && test ! -L "$path"
done
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT" && test ! -L "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')"
test "$gpu_name" = "NVIDIA H100 PCIe"

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -P -s -B -m sai.data.learnability_score \
  --geometry "$GEOMETRY" \
  --geometry-sha256 "$GEOMETRY_SHA256" \
  --family "$FAMILY" \
  --scale "$SCALE" \
  --probe-result "$PROBE_RESULT" \
  --probe-result-sha256 "$PROBE_RESULT_SHA256" \
  --strong-checkpoint "$STRONG_CHECKPOINT" \
  --strong-checkpoint-manifest "$STRONG_CHECKPOINT_MANIFEST" \
  --weak-milestone "$WEAK_MILESTONE" \
  --weak-milestone-step "$WEAK_MILESTONE_STEP" \
  --probe-training-stream "$PROBE_TRAINING_STREAM" \
  --probe-training-stream-identity "$PROBE_TRAINING_STREAM_IDENTITY" \
  --target-stream "$TARGET_STREAM" \
  --target-stream-identity "$TARGET_STREAM_IDENTITY" \
  --runtime-receipt "$RUNTIME_RECEIPT" \
  --runtime-receipt-sha256 "$RUNTIME_RECEIPT_SHA256" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT"

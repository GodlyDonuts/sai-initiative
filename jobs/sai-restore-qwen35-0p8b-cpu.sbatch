#!/bin/bash
#SBATCH --job-name=sai-restore-qwen08
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for path in "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$OUTPUT_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$OUTPUT_ROOT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

scratch_parent="${SLURM_TMPDIR:-/tmp}"
test -d "$scratch_parent"
test ! -L "$scratch_parent"
available_bytes="$(( $(df -Pk "$scratch_parent" | awk 'NR==2 {print $4}') * 1024 ))"
(( available_bytes >= 4_000_000_000 ))
umask 077
cache_root="$(mktemp -d "$scratch_parent/sai-qwen08-$SLURM_JOB_ID.XXXXXX")"
case "$cache_root" in "$scratch_parent"/sai-qwen08-"$SLURM_JOB_ID".??????) ;; *) exit 2 ;; esac
test "$(stat -c %u "$cache_root")" = "$(id -u)"
test "$(stat -c %a "$cache_root")" = 700
cleanup() {
  case "$cache_root" in "$scratch_parent"/sai-qwen08-"$SLURM_JOB_ID".??????) ;; *) return 2 ;; esac
  if test -d "$cache_root" && test ! -L "$cache_root"; then
    find "$cache_root" -xdev -depth -delete
  fi
}
trap cleanup EXIT

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1
"$PYTHON" -P -s -B -m sai.data.hf_model_snapshot \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$cache_root"
cleanup
trap - EXIT

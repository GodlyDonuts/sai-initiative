#!/bin/bash
#SBATCH --job-name=sai-curriculum-500m
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${DECONTAMINATION_JOB_ID:?DECONTAMINATION_JOB_ID is required}"
: "${DECONTAMINATED_SOURCE:?DECONTAMINATED_SOURCE is required}"
: "${DECONTAMINATION_RECEIPT:?DECONTAMINATION_RECEIPT is required}"
: "${CURRICULUM_OUTPUT:?CURRICULUM_OUTPUT is required}"
: "${CURRICULUM_RECEIPT:?CURRICULUM_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in "$SAI_ROOT" "$PYTHON" "$DECONTAMINATED_SOURCE" \
  "$DECONTAMINATION_RECEIPT" "$CURRICULUM_OUTPUT" \
  "$CURRICULUM_RECEIPT" "$LOG_ROOT"; do
  require_absolute "$path"
done

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU curriculum builder was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$DECONTAMINATED_SOURCE"
test ! -L "$DECONTAMINATED_SOURCE"
test -f "$DECONTAMINATION_RECEIPT"
test ! -L "$DECONTAMINATION_RECEIPT"
test ! -e "$CURRICULUM_OUTPUT"
test ! -L "$CURRICULUM_OUTPUT"
test ! -e "$CURRICULUM_RECEIPT"
test ! -L "$CURRICULUM_RECEIPT"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"

state="$(sacct -X -j "$DECONTAMINATION_JOB_ID" -n -P -o State | sed -n '1p')"
test "$state" = "COMPLETED"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -P -s -B -m sai.data.curriculum build \
  --source "$DECONTAMINATED_SOURCE" \
  --decontamination-receipt "$DECONTAMINATION_RECEIPT" \
  --output "$CURRICULUM_OUTPUT" \
  --receipt "$CURRICULUM_RECEIPT" \
  --minimum-documents-per-band 10000 \
  --workers "$SLURM_CPUS_PER_TASK"

"$PYTHON" -P -s -B -m sai.data.curriculum validate \
  --receipt "$CURRICULUM_RECEIPT"

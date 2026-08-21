#!/bin/bash
#SBATCH --job-name=sai-decontam-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_decontaminate_%j.out
#SBATCH --error=logs/sai_decontaminate_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SOURCE:?SOURCE is required}"
: "${BOUNDARIES:?BOUNDARIES is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${RECEIPT:?RECEIPT is required}"

test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$SOURCE"
test ! -e "$OUTPUT"
test ! -e "$RECEIPT"

IFS=: read -r -a boundary_paths <<< "$BOUNDARIES"
test "${#boundary_paths[@]}" = 20
boundary_args=()
for path in "${boundary_paths[@]}"; do
  test -f "$path"
  test ! -L "$path"
  boundary_args+=(--boundary "$path")
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec "$PYTHON" -P -s -B -m sai.data.decontamination build \
  --source "$SOURCE" \
  "${boundary_args[@]}" \
  --output "$OUTPUT" \
  --receipt "$RECEIPT"

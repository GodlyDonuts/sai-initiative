#!/bin/bash
#SBATCH --job-name=sai-compose-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

required=(
  SAI_ROOT EXPECTED_COMMIT PYTHON LOG_ROOT PARENT_STREAM SCORES TAXONOMY
  CURRICULUM_RECEIPT ANNOTATIONS PROGRESSION_REPORT OUTPUT
)
for name in "${required[@]}"; do
  test -n "${!name:-}" || { echo "$name is required" >&2; exit 2; }
done
WORKERS="${WORKERS:-16}"
case "$WORKERS" in ''|*[!0-9]*|0) exit 2 ;; esac

require_absolute() {
  case "$1" in /*) ;; *) echo "path must be absolute: $1" >&2; exit 2 ;; esac
}
for path in \
  "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$PARENT_STREAM" "$SCORES" \
  "$TAXONOMY" "$CURRICULUM_RECEIPT" "$ANNOTATIONS" \
  "$PROGRESSION_REPORT" "$OUTPUT"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"
for path in "$PARENT_STREAM" "$SCORES"; do
  test -d "$path" && test ! -L "$path"
done
for path in "$TAXONOMY" "$CURRICULUM_RECEIPT" "$ANNOTATIONS" "$PROGRESSION_REPORT"; do
  test -f "$path" && test ! -L "$path"
done
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT" && test ! -L "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *)
  echo "CPU curriculum composer was exposed to a GPU" >&2
  exit 2
;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON" -P -s -B -m sai.data.semantic_learnability_curriculum build \
  --parent-stream "$PARENT_STREAM" \
  --scores "$SCORES" \
  --taxonomy "$TAXONOMY" \
  --curriculum-receipt "$CURRICULUM_RECEIPT" \
  --annotations "$ANNOTATIONS" \
  --progression-report "$PROGRESSION_REPORT" \
  --output "$OUTPUT" \
  --workers "$WORKERS"

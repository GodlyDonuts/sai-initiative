#!/bin/bash
#SBATCH --job-name=sai-screen-mc-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${EVALUATION_JOB_IDS:?EVALUATION_JOB_IDS is required}"
: "${COMPARISON_ARGUMENTS:?COMPARISON_ARGUMENTS is required}"

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU comparison job was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"

read -r -a evaluation_job_ids <<< "$EVALUATION_JOB_IDS"
test "${#evaluation_job_ids[@]}" = 6
test "$(printf '%s\n' "${evaluation_job_ids[@]}" | sort -u | wc -l | tr -d ' ')" = 6
for job_id in "${evaluation_job_ids[@]}"; do
  [[ "$job_id" =~ ^[0-9]+$ ]]
  state="$(sacct -j "$job_id" -X -n -P -o State | sed -n '1p')"
  exit_code="$(sacct -j "$job_id" -X -n -P -o ExitCode | sed -n '1p')"
  restarts="$(sacct -j "$job_id" -X -n -P -o Restarts | sed -n '1p')"
  test "$state" = "COMPLETED"
  test "$exit_code" = "0:0"
  test "$restarts" = "0"
done

read -r -a comparison_arguments <<< "$COMPARISON_ARGUMENTS"
test "${#comparison_arguments[@]}" = 12
export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -P -s -B -m sai.evaluation.short_screen_compare \
  "${comparison_arguments[@]}" --output "$OUTPUT"
test -f "$OUTPUT"
test ! -L "$OUTPUT"

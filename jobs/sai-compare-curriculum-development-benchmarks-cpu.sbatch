#!/bin/bash
#SBATCH --job-name=sai-compare-curriculum-boards
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${ORDER_COMPARISON_JOB_ID:?ORDER_COMPARISON_JOB_ID is required}"
: "${ORDER_COMPARISON:?ORDER_COMPARISON is required}"
: "${CURRICULUM_MMLU_JOB_ID:?CURRICULUM_MMLU_JOB_ID is required}"
: "${CURRICULUM_MMLU:?CURRICULUM_MMLU is required}"
: "${CURRICULUM_MUSR_JOB_ID:?CURRICULUM_MUSR_JOB_ID is required}"
: "${CURRICULUM_MUSR:?CURRICULUM_MUSR is required}"
: "${CONTROL_MMLU_JOB_ID:?CONTROL_MMLU_JOB_ID is required}"
: "${CONTROL_MMLU:?CONTROL_MMLU is required}"
: "${CONTROL_MUSR_JOB_ID:?CONTROL_MUSR_JOB_ID is required}"
: "${CONTROL_MUSR:?CONTROL_MUSR is required}"
: "${H100_EVALUATION_JOB_IDS:?H100_EVALUATION_JOB_IDS is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for value in "$ORDER_COMPARISON_JOB_ID" "$CURRICULUM_MMLU_JOB_ID" \
  "$CURRICULUM_MUSR_JOB_ID" "$CONTROL_MMLU_JOB_ID" "$CONTROL_MUSR_JOB_ID"; do
  [[ "$value" =~ ^[0-9]+$ ]]
done
read -r -a h100_job_ids <<< "$H100_EVALUATION_JOB_IDS"
test "${#h100_job_ids[@]}" = 18
test "$(printf '%s\n' "${h100_job_ids[@]}" | sort -u | wc -l | tr -d ' ')" = 18
for path in "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$ORDER_COMPARISON" \
  "$CURRICULUM_MMLU" "$CURRICULUM_MUSR" "$CONTROL_MMLU" \
  "$CONTROL_MUSR" "$OUTPUT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "benchmark comparator was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
for path in "$ORDER_COMPARISON" "$CURRICULUM_MMLU" "$CURRICULUM_MUSR" \
  "$CONTROL_MMLU" "$CONTROL_MUSR"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$(dirname "$OUTPUT")"
test ! -L "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

for job_id in "$ORDER_COMPARISON_JOB_ID" "$CURRICULUM_MMLU_JOB_ID" \
  "$CURRICULUM_MUSR_JOB_ID" "$CONTROL_MMLU_JOB_ID" "$CONTROL_MUSR_JOB_ID"; do
  row="$(sacct -X -j "$job_id" -n -P -o State,ExitCode,Restarts | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done
for job_id in "${h100_job_ids[@]}"; do
  row="$(sacct -X -j "$job_id" -n -P -o State,ExitCode,Restarts | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec "$PYTHON" -P -s -B -m sai.evaluation.curriculum_benchmark_compare \
  --order-comparison "$ORDER_COMPARISON" \
  --curriculum-mmlu-pro "$CURRICULUM_MMLU" \
  --curriculum-musr "$CURRICULUM_MUSR" \
  --control-mmlu-pro "$CONTROL_MMLU" \
  --control-musr "$CONTROL_MUSR" \
  --output "$OUTPUT"

#!/bin/bash
#SBATCH --job-name=sai-authored-model-review
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc36,evc43,evc50

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${SAI_ROOT:?}"
: "${EXPECTED_COMMIT:?}"
: "${PYTHON:?}"
: "${REVIEWER:?}"
: "${MODEL_ROOT:?}"
: "${MODEL_MANIFEST:?}"
: "${MODEL_RECEIPT:?}"
: "${REVIEW_PACKET:?}"
: "${REVIEW_PACKET_RECEIPT:?}"
: "${REVIEW_PACKET_SHA256:?}"
: "${REVIEW_PACKET_RECEIPT_SHA256:?}"
: "${CONCEPT_LIST:?}"
: "${ANNOTATION_POLICY:?}"
: "${OUTPUT_ROOT:?}"

[[ "$REVIEWER" == qwen35_9b || "$REVIEWER" == smollm3_3b ]]
[[ "$OUTPUT_ROOT" = /* && "$OUTPUT_ROOT" != / && "$OUTPUT_ROOT" != "$HOME" ]]
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ]]
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

exec "$PYTHON" -P -s -B -m sai.data.authored_review_model \
  generate \
  --reviewer "$REVIEWER" \
  --model-root "$MODEL_ROOT" \
  --manifest "$MODEL_MANIFEST" \
  --restoration-receipt "$MODEL_RECEIPT" \
  --review-packet "$REVIEW_PACKET" \
  --review-packet-receipt "$REVIEW_PACKET_RECEIPT" \
  --expected-review-packet-sha256 "$REVIEW_PACKET_SHA256" \
  --expected-review-packet-receipt-sha256 "$REVIEW_PACKET_RECEIPT_SHA256" \
  --concept-list "$CONCEPT_LIST" \
  --annotation-policy "$ANNOTATION_POLICY" \
  --output-root "$OUTPUT_ROOT"

#!/bin/bash
#SBATCH --job-name=sai-authored-review-context
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GIT_OPTIONAL_LOCKS=0

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
: "${OUTPUT:?}"

export PYTHONPATH="$SAI_ROOT/src"
[[ "$REVIEWER" == qwen35_9b || "$REVIEWER" == smollm3_3b ]]
[[ "$OUTPUT" = /* && "$OUTPUT" != / && "$OUTPUT" != "$HOME" ]]
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"

exec "$PYTHON" -P -s -B -m sai.data.authored_review_context \
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
  --output "$OUTPUT"

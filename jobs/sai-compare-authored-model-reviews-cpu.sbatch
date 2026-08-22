#!/bin/bash
#SBATCH --job-name=sai-compare-authored-reviews
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GIT_OPTIONAL_LOCKS=0

for name in SAI_ROOT EXPECTED_COMMIT PYTHON QWEN_MODEL_ROOT QWEN_MANIFEST \
  QWEN_MODEL_RECEIPT QWEN_REVIEW_ROOT SMOL_MODEL_ROOT SMOL_MANIFEST \
  SMOL_MODEL_RECEIPT SMOL_REVIEW_ROOT REVIEW_PACKET REVIEW_PACKET_RECEIPT \
  REVIEW_PACKET_SHA256 REVIEW_PACKET_RECEIPT_SHA256 CONCEPT_LIST \
  ANNOTATION_POLICY OUTPUT; do
  [[ -n "${!name:-}" ]]
done
export PYTHONPATH="$SAI_ROOT/src"
[[ "$OUTPUT" = /* && "$OUTPUT" != / && "$OUTPUT" != "$HOME" ]]
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"

exec "$PYTHON" -P -s -B -m sai.data.authored_review_model_compare \
  compare \
  --qwen-model-root "$QWEN_MODEL_ROOT" \
  --qwen-manifest "$QWEN_MANIFEST" \
  --qwen-restoration-receipt "$QWEN_MODEL_RECEIPT" \
  --qwen-review-root "$QWEN_REVIEW_ROOT" \
  --smol-model-root "$SMOL_MODEL_ROOT" \
  --smol-manifest "$SMOL_MANIFEST" \
  --smol-restoration-receipt "$SMOL_MODEL_RECEIPT" \
  --smol-review-root "$SMOL_REVIEW_ROOT" \
  --review-packet "$REVIEW_PACKET" \
  --review-packet-receipt "$REVIEW_PACKET_RECEIPT" \
  --expected-review-packet-sha256 "$REVIEW_PACKET_SHA256" \
  --expected-review-packet-receipt-sha256 "$REVIEW_PACKET_RECEIPT_SHA256" \
  --concept-list "$CONCEPT_LIST" \
  --annotation-policy "$ANNOTATION_POLICY" \
  --output "$OUTPUT"

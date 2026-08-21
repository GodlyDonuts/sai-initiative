#!/bin/bash
#SBATCH --job-name=sai-fla-parity
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc35,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_fla_parity_%j.out
#SBATCH --error=logs/sai_fla_parity_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test ! -e "$OUTPUT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -P -s -B - <<'PY'
import importlib.metadata
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
probe = torch.ones(128, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(probe.square().sum())
assert importlib.metadata.version("flash-linear-attention") == "0.4.2"
PY

exec "$PYTHON" -P -s -B -m sai.training.fla_parity --output "$OUTPUT"

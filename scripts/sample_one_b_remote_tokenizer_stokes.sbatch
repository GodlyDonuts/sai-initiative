#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-tokenizer-remote
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-remote-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-remote-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
: "${SAI_TOKENIZER_COMPONENT:?code or pleias is required}"
[[ "${SAI_TOKENIZER_COMPONENT}" == code || "${SAI_TOKENIZER_COMPONENT}" == pleias ]]
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
if [[ "${SAI_TOKENIZER_COMPONENT}" == code ]]; then
  sai_admission=${sai_root}/common-pile-stack-edu-practical-admission-20260826-r1
  sai_bytes=128000000
  sai_parents=1
else
  sai_admission=${sai_root}/pleias-practical-admission-20260826-r3
  sai_bytes=256000000
  sai_parents=4
fi
python -m sai.data.one_b_tokenizer_sample "${SAI_TOKENIZER_COMPONENT}" \
  --admission-root "${sai_admission}" \
  --locator-shard 0 \
  --maximum-jsonl-bytes "${sai_bytes}" \
  --maximum-parents "${sai_parents}" \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}" \
  --output-root "${sai_root}/one-b-tokenizer-sample-20260826-r1/${SAI_TOKENIZER_COMPONENT}"

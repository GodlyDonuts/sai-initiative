#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-hf-git-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-git-repair-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-git-repair-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
: "${HF_TOKEN:?Hugging Face token is required}"
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export GIT_LFS_SKIP_SMUDGE=1

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
admission_root="${sai_root}/pleias-practical-admission-20260826-r3"
publish_root="${sai_root}/pleias-practical-hf-publish-20260826-r3"
books_root="${sai_root}/institutional-books-practical-admission-20260826-r1"
aggregate_path="${publish_root}/aggregate.json"
metadata_path="${publish_root}/metadata-publication.json"
audit_path="${sai_root}/sai_evidence/practical-corpus-readiness/20260826-r3/receipt.json"
missing_shards=(107 114 116 123 124 127)

[[ ! -e "${aggregate_path}" ]]
[[ ! -e "${metadata_path}" ]]
[[ ! -e "${audit_path}" ]]
for shard in "${missing_shards[@]}"; do
  [[ ! -e "${publish_root}/shards/shard_$(printf '%05d' "${shard}")" ]]
done

scratch="$(mktemp -d "${TMPDIR:-/tmp}/sai-hf-git-repair.XXXXXX")"
cleanup() {
  rm -rf -- "${scratch}"
}
trap cleanup EXIT
repository="${scratch}/Sai"
git clone --quiet --depth 1 https://huggingface.co/datasets/Godlydonuts/Sai.git "${repository}"
git -C "${repository}" config user.name Sai-Compiler
git -C "${repository}" config user.email sai-compiler@users.noreply.github.com
git -C "${repository}" lfs install --local >/dev/null

for shard in "${missing_shards[@]}"; do
  formatted="$(printf '%05d' "${shard}")"
  destination="${repository}/training/practical/pleias/20260826-r3/shards/shard_${formatted}/locators.parquet"
  mkdir -p "$(dirname "${destination}")"
  cp -- "${admission_root}/shards/shard_${formatted}/locators.parquet" "${destination}"
done
git -C "${repository}" add training/practical/pleias/20260826-r3/shards
git -C "${repository}" commit --quiet -m 'Repair six verified practical locator shards'

basic_auth="$(printf 'Godlydonuts:%s' "${HF_TOKEN}" | base64 -w0)"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.extraHeader
export GIT_CONFIG_VALUE_0="Authorization: Basic ${basic_auth}"
git -C "${repository}" push --quiet origin HEAD:main
unset basic_auth GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
locator_revision="$(git -C "${repository}" rev-parse HEAD)"

record_args=()
for shard in "${missing_shards[@]}"; do
  record_args+=(--shard-index "${shard}")
done
python -m sai.data.practical_hf_publish record-shards \
  --admission-root "${admission_root}" \
  --publish-root "${publish_root}" \
  --logical-shards 128 \
  --revision "${locator_revision}" \
  --token-env HF_TOKEN \
  "${record_args[@]}"

python -m sai.data.practical_hf_publish aggregate \
  --admission-root "${admission_root}" \
  --publish-root "${publish_root}" \
  --logical-shards 128 \
  --output "${aggregate_path}" \
  --token-env HF_TOKEN

books_manifest="$(python - "${books_root}/receipt.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest"]["path"])
PY
)"
declare -a metadata_sources=(
  "${books_root}/receipt.json|training/practical/metadata/20260826-r3/books/receipt.lfs.json"
  "${books_root}/${books_manifest}|training/practical/metadata/20260826-r3/books/manifest.lfs.jsonl"
  "${admission_root}/receipt.json|training/practical/metadata/20260826-r3/pleias/receipt.lfs.json"
  "${aggregate_path}|training/practical/metadata/20260826-r3/pleias/publish-aggregate.lfs.json"
)
for mapping in "${metadata_sources[@]}"; do
  source_path="${mapping%%|*}"
  remote_path="${mapping#*|}"
  mkdir -p "${repository}/$(dirname "${remote_path}")"
  cp -- "${source_path}" "${repository}/${remote_path}"
done
git -C "${repository}" add training/practical/metadata/20260826-r3
git -C "${repository}" commit --quiet -m 'Publish verified practical corpus metadata'
basic_auth="$(printf 'Godlydonuts:%s' "${HF_TOKEN}" | base64 -w0)"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.extraHeader
export GIT_CONFIG_VALUE_0="Authorization: Basic ${basic_auth}"
git -C "${repository}" push --quiet origin HEAD:main
unset basic_auth GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
metadata_revision="$(git -C "${repository}" rev-parse HEAD)"

python -m sai.data.practical_hf_publish record-metadata \
  --books-root "${books_root}" \
  --pleias-root "${admission_root}" \
  --publish-aggregate "${aggregate_path}" \
  --output "${metadata_path}" \
  --revision "${metadata_revision}" \
  --token-env HF_TOKEN

mkdir -p "$(dirname "${audit_path}")"
python -m sai.data.practical_corpus_audit \
  --books-receipt "${books_root}/receipt.json" \
  --pleias-receipt "${admission_root}/receipt.json" \
  --publication-receipt "${metadata_path}" \
  --output "${audit_path}" \
  --minimum-combined-text-bytes 1900000000000 \
  --maximum-combined-text-bytes 2000000000000

printf 'locator_revision=%s metadata_revision=%s audit=%s\n' \
  "${locator_revision}" "${metadata_revision}" "${audit_path}"

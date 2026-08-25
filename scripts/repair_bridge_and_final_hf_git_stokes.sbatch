#!/usr/bin/env bash
#SBATCH --job-name=sai-final-hf-git-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/final-hf-git-repair-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/final-hf-git-repair-%j.err

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
admission_root="${sai_root}/bridge-training-component-admission-20260826-r1"
bridge_publication="${sai_root}/sai_evidence/bridge-training-component-hf-publication/20260826-r1/receipt.json"
release_root="${sai_root}/final-training-release-20260826-r1"
release_path="${release_root}/release.json"
release_publication="${release_root}/hf-publication-receipt.json"
[[ ! -e "${bridge_publication}" ]]
[[ ! -e "${release_path}" ]]
[[ ! -e "${release_publication}" ]]

scratch="$(mktemp -d "${TMPDIR:-/tmp}/sai-final-hf-repair.XXXXXX")"
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
repository="${scratch}/Sai"
git clone --quiet --depth 1 https://huggingface.co/datasets/Godlydonuts/Sai.git "${repository}"
git -C "${repository}" config user.name Sai-Compiler
git -C "${repository}" config user.email sai-compiler@users.noreply.github.com
git -C "${repository}" lfs install --local >/dev/null

preupload_and_push() {
  local source_path=$1
  local remote_path=$2
  local message=$3
  python - "${source_path}" "${remote_path}" <<'PY'
import os
import sys
from huggingface_hub import CommitOperationAdd, HfApi

HfApi(token=os.environ["HF_TOKEN"]).preupload_lfs_files(
    repo_id="Godlydonuts/Sai",
    additions=[CommitOperationAdd(path_in_repo=sys.argv[2], path_or_fileobj=sys.argv[1])],
    repo_type="dataset",
    revision="main",
    num_threads=1,
)
PY
  mkdir -p "${repository}/$(dirname "${remote_path}")"
  cp -- "${source_path}" "${repository}/${remote_path}"
  git -C "${repository}" add "${remote_path}"
  git -C "${repository}" commit --quiet -m "${message}"
}

preupload_and_push "${admission_root}/train.jsonl.gz" \
  training/final/cross-domain-connections/20260826-r1/train.jsonl.gz \
  'Publish admitted cross-domain connection component'
preupload_and_push "${admission_root}/receipt.json" \
  training/final/cross-domain-connections/20260826-r1/receipt.lfs.json \
  'Publish cross-domain connection admission receipt'
basic_auth="$(printf 'Godlydonuts:%s' "${HF_TOKEN}" | base64 -w0)"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader
export GIT_CONFIG_VALUE_0="Authorization: Basic ${basic_auth}"
git -C "${repository}" push --quiet origin HEAD:main
unset basic_auth GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
bridge_revision="$(git -C "${repository}" rev-parse HEAD)"

python -m sai.data.bridge_component_hf_publish \
  --admission-root "${admission_root}" --output "${bridge_publication}" \
  --revision "${bridge_revision}" --token-env HF_TOKEN

mkdir -p "${release_root}"
python -m sai.data.final_training_release \
  --foundation-audit "${sai_root}/sai_evidence/practical-corpus-readiness/20260826-r3/receipt.json" \
  --code-admission "${sai_root}/common-pile-stack-edu-practical-admission-20260826-r1/receipt.json" \
  --code-publication "${sai_root}/common-pile-stack-edu-practical-hf-publish-20260826-r1/metadata-publication.json" \
  --bridge-admission "${admission_root}/receipt.json" \
  --bridge-publication "${bridge_publication}" \
  --output "${release_path}"

git -C "${repository}" fetch --quiet origin main
git -C "${repository}" reset --hard --quiet origin/main
preupload_and_push "${release_path}" \
  training/final/metadata/20260826-r1/release.lfs.json \
  'Publish signed Sai training-data release'
basic_auth="$(printf 'Godlydonuts:%s' "${HF_TOKEN}" | base64 -w0)"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader
export GIT_CONFIG_VALUE_0="Authorization: Basic ${basic_auth}"
git -C "${repository}" push --quiet origin HEAD:main
unset basic_auth GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
release_revision="$(git -C "${repository}" rev-parse HEAD)"

python -m sai.data.final_training_release_hf_publish \
  --release "${release_path}" --output "${release_publication}" \
  --revision "${release_revision}" --token-env HF_TOKEN
printf 'bridge_revision=%s release_revision=%s release=%s\n' \
  "${bridge_revision}" "${release_revision}" "${release_path}"

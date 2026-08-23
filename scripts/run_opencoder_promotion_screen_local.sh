#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"
export PYTHONPATH=src

sai_judgments=artifacts/sai_opencoder_code_web_audit_20260826_r1/judgments
sai_output=artifacts/sai_opencoder_code_web_promotion_screen_20260826_r1.json
sai_screen_shards=({64..71} {96..103})

for sai_wait_index in {1..34560}; do
  sai_complete=0
  for shard_index in "${sai_screen_shards[@]}"; do
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      sai_complete=$((sai_complete + 1))
    fi
  done
  if [[ "${sai_complete}" = 16 ]]; then
    break
  fi
  sleep 5
done

if [[ -e "${sai_output}" ]]; then
  echo 'OpenCoder promotion screen already exists; refusing duplicate' >&2
  exit 1
fi
PYTHONPATH=src python3 -m sai.data.opencoder_promotion_screen \
  --population-root artifacts/sai_opencoder_code_web_audit_20260826_r1 \
  --judgments-root "${sai_judgments}" \
  --output "${sai_output}"

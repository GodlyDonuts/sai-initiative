#!/bin/bash

set -euo pipefail

: "${SAI_SHARD_INDICES:?SAI_SHARD_INDICES is required}"

sai_repo=/lustre/fs1/home/sa305415/sai-initiative
sai_census=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-20260826-r2
sai_recovery=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-recovery-20260826-r1
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/pleias-metadata-census-incomplete-originals/20260826-r1
sai_original_array=818243
sai_dispatch_job=818603
sai_aggregate_job=818244
sai_receipt=${sai_recovery}/accelerated-dispatch-$(date -u +%Y%m%dT%H%M%SZ).jsonl

mkdir -p "${sai_recovery}" "${sai_evidence}"
: > "${sai_receipt}.partial"

sai_merge_jobs=()
for sai_shard_index in ${SAI_SHARD_INDICES}; do
  if [[ ! "${sai_shard_index}" =~ ^[0-9]+$ ]] \
      || (( sai_shard_index < 0 || sai_shard_index > 127 )); then
    echo "invalid PleIAs census shard: ${sai_shard_index}" >&2
    exit 1
  fi
  sai_shard=$(printf '%s/shards/shard_%05d' "${sai_census}" "${sai_shard_index}")
  if [[ -f "${sai_shard}/receipt.json" ]]; then
    printf '{"action":"skip_complete","shard_index":%d}\n' \
      "${sai_shard_index}" >> "${sai_receipt}.partial"
    continue
  fi
  sai_recovery_shard=$(printf '%s/shard_%05d' "${sai_recovery}" "${sai_shard_index}")
  sai_dispatch_marker=${sai_recovery_shard}/dispatch.json
  if [[ -f "${sai_dispatch_marker}" ]]; then
    sai_merge_job=$(python3 - "${sai_dispatch_marker}" "${sai_shard_index}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    row = json.load(handle)
if (
    row.get("schema") != "sai-pleias-metadata-census-recovery-dispatch-v1"
    or row.get("shard_index") != int(sys.argv[2])
    or not str(row.get("merge_job", "")).isdigit()
):
    raise SystemExit("recovery dispatch marker differs")
print(row["merge_job"])
PY
    )
    sai_merge_jobs+=("${sai_merge_job}")
    printf '{"action":"reuse_dispatch","merge_job":"%s","shard_index":%d}\n' \
      "${sai_merge_job}" "${sai_shard_index}" >> "${sai_receipt}.partial"
    continue
  fi
  if [[ -e "${sai_recovery_shard}" ]]; then
    echo "unreceipted recovery shard already exists: ${sai_recovery_shard}" >&2
    exit 1
  fi

  sai_original_job=${sai_original_array}_${sai_shard_index}
  if squeue -h -j "${sai_original_job}" | grep -q .; then
    scancel "${sai_original_job}" || true
  fi
  for _sai_wait in $(seq 1 120); do
    if ! squeue -h -j "${sai_original_job}" | grep -q .; then
      break
    fi
    sleep 1
  done
  if squeue -h -j "${sai_original_job}" | grep -q .; then
    echo "original shard did not terminate: ${sai_original_job}" >&2
    exit 1
  fi
  if [[ -f "${sai_shard}/receipt.json" ]]; then
    printf '{"action":"completed_during_cancel","original_job":"%s","shard_index":%d}\n' \
      "${sai_original_job}" "${sai_shard_index}" >> "${sai_receipt}.partial"
    continue
  fi
  if [[ -e "${sai_shard}" ]]; then
    sai_incomplete=$(printf '%s/shard_%05d' "${sai_evidence}" "${sai_shard_index}")
    if [[ -e "${sai_incomplete}" ]]; then
      echo "incomplete evidence target already exists: ${sai_incomplete}" >&2
      exit 1
    fi
    mv "${sai_shard}" "${sai_incomplete}"
  fi

  mkdir -p "${sai_recovery_shard}"
  sai_segment_job=$(sbatch --parsable \
    --array=0-7%8 \
    --export=ALL,SAI_SHARD_INDEX="${sai_shard_index}" \
    "${sai_repo}/scripts/recover_pleias_metadata_census_segment_stokes.sbatch")
  if ! sai_merge_job=$(sbatch --parsable \
      --dependency=afterok:"${sai_segment_job}" \
      --export=ALL,SAI_SHARD_INDEX="${sai_shard_index}" \
      "${sai_repo}/scripts/merge_pleias_metadata_census_recovery_stokes.sbatch"); then
    scancel "${sai_segment_job}"
    exit 1
  fi
  python3 - "${sai_shard_index}" "${sai_segment_job}" "${sai_merge_job}" \
      > "${sai_dispatch_marker}.partial" <<'PY'
import json
import sys

print(json.dumps({
    "schema": "sai-pleias-metadata-census-recovery-dispatch-v1",
    "shard_index": int(sys.argv[1]),
    "segment_job": sys.argv[2],
    "merge_job": sys.argv[3],
}, sort_keys=True))
PY
  mv "${sai_dispatch_marker}.partial" "${sai_dispatch_marker}"
  sai_merge_jobs+=("${sai_merge_job}")
  printf '{"action":"accelerated_recovery","merge_job":"%s","original_job":"%s","segment_job":"%s","shard_index":%d}\n' \
    "${sai_merge_job}" "${sai_original_job}" "${sai_segment_job}" \
    "${sai_shard_index}" >> "${sai_receipt}.partial"
done

mv "${sai_receipt}.partial" "${sai_receipt}"

if (( ${#sai_merge_jobs[@]} > 0 )); then
  sai_dependency=$(IFS=:; echo "${sai_dispatch_job}:${sai_merge_jobs[*]}")
  scontrol update JobId="${sai_aggregate_job}" \
    Dependency=afterok:"${sai_dependency}"
fi

printf '{"accelerated_merge_jobs":%d,"aggregate_job":"%s","dispatch_job":"%s","receipt":"%s"}\n' \
  "${#sai_merge_jobs[@]}" "${sai_aggregate_job}" "${sai_dispatch_job}" \
  "${sai_receipt}"

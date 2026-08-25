#!/usr/bin/env bash

# Activate one immutable, shared Slurm configuration and prove that it targets
# the intended cluster.  SLURM_CONF_SERVER cannot switch clusters from a node
# that already has a local slurm.conf, so cross-cluster launchers must use an
# explicit, hash-bound configuration file instead.
sai_activate_verified_slurm_config() {
  local config_path="$1"
  local expected_sha256="$2"
  local expected_cluster="$3"
  local actual_sha256
  local actual_cluster
  local cluster_query_attempt

  [[ "${config_path}" == /* ]] || return 2
  [[ "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]] || return 2
  case "${expected_cluster}" in
    newton|stokes) ;;
    *) return 2 ;;
  esac
  [[ -f "${config_path}" && ! -L "${config_path}" ]] || return 2
  [[ "$(stat -c '%h' -- "${config_path}")" == 1 ]] || return 2
  actual_sha256="$(sha256sum -- "${config_path}" | awk '{print $1}')" || return 2
  [[ "${actual_sha256}" == "${expected_sha256}" ]] || return 2

  export SLURM_CONF="${config_path}"
  # Batch jobs inherit the submitting cluster name. Leaving `stokes` here
  # suppresses Newton's ClusterName even when SLURM_CONF points at Newton.
  unset SLURM_CONF_SERVER SLURM_CLUSTER_NAME
  actual_cluster=
  # Retry only this read-only identity query; never retry a submission or
  # accept an empty/mismatched cluster name.  The parser deliberately consumes
  # all output: exiting after ClusterName would give `scontrol` SIGPIPE under
  # the launcher's `pipefail` and discard an otherwise valid result.
  for cluster_query_attempt in 1 2 3 4 5; do
    actual_cluster="$(
      scontrol show config 2>/dev/null | awk -F= '
        /^ClusterName/ {
          gsub(/[[:space:]]/, "", $2)
          print $2
        }
      '
    )" || actual_cluster=
    [[ -n "${actual_cluster}" ]] && break
    sleep 1
  done
  [[ "${actual_cluster}" == "${expected_cluster}" ]] || return 2
}

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

  [[ "${config_path}" == /* ]]
  [[ "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]]
  case "${expected_cluster}" in
    newton|stokes) ;;
    *) return 2 ;;
  esac
  [[ -f "${config_path}" && ! -L "${config_path}" ]]
  [[ "$(stat -c '%h' -- "${config_path}")" == 1 ]]
  actual_sha256="$(sha256sum -- "${config_path}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "${expected_sha256}" ]]

  export SLURM_CONF="${config_path}"
  unset SLURM_CONF_SERVER
  actual_cluster="$(
    scontrol show config | awk -F= '
      /^ClusterName/ {
        gsub(/[[:space:]]/, "", $2)
        print $2
        exit
      }
    '
  )"
  [[ "${actual_cluster}" == "${expected_cluster}" ]]
}

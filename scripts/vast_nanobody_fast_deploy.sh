#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
onstart_script="${script_dir}/nanobody_cache_onstart.sh"

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 volumes" \
    "  $0 gpu-offers [QUERY]" \
    "  $0 volume-offers MACHINE_ID [SIZE_GB]" \
    "  $0 volume-create VOLUME_OFFER_ID [SIZE_GB] [NAME]" \
    "  $0 offers VOLUME_ID [EXTRA_QUERY]" \
    "  $0 launch GPU_OFFER_ID VOLUME_ID IMAGE_REF [DISK_GB]" \
    "" \
    "Mutating commands are dry runs unless NOVA_CONFIRM_SPEND=1 is set." \
    "IMAGE_REF must use an immutable non-latest tag or, preferably, a digest."
}

require_tools() {
  command -v vastai >/dev/null
  command -v jq >/dev/null
}

json_array() {
  jq 'if type == "array" then . elif has("offers") then .offers elif has("volumes") then .volumes else [] end'
}

offer_json_on_machine() {
  local offer_id=$1
  local machine_id=$2
  vastai search offers "machine_id=${machine_id} rentable=true" --raw \
    | json_array \
    | jq -ec --argjson id "$offer_id" 'map(select(.id == $id)) | first'
}

volume_json() {
  local volume_id=$1
  vastai show volumes --raw \
    | json_array \
    | jq -ec --argjson id "$volume_id" 'map(select(.id == $id)) | first'
}

run_billable() {
  if [[ "${NOVA_CONFIRM_SPEND:-0}" != "1" ]]; then
    printf '%s\n' "Dry run; set NOVA_CONFIRM_SPEND=1 to execute:" >&2
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

require_tools
command=${1:-}

case "$command" in
  volumes)
    vastai show volumes
    ;;

  gpu-offers)
    [[ $# -le 2 ]] || { usage >&2; exit 2; }
    query=${2:-'num_gpus=8 gpu_name=RTX_4090 reliability>0.95 disk_space>=80 rentable=true'}
    vastai search offers "$query" --raw \
      | json_array \
      | jq -r '["OFFER_ID", "MACHINE_ID", "GPU", "GPUS", "DPH", "RELIABILITY", "DOWN_MBPS"],
               (.[] | [.id, .machine_id, .gpu_name, .num_gpus, .dph_total, .reliability, .inet_down])
               | @tsv'
    ;;

  volume-offers)
    [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
    machine_id=$2
    size_gb=${3:-30}
    [[ "$machine_id" =~ ^[0-9]+$ && "$size_gb" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
    vastai search volumes "machine_id=${machine_id} disk_space>=${size_gb} reliability>0.95" \
      --storage "$size_gb" -o 'storage_cost'
    ;;

  volume-create)
    [[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
    volume_offer_id=$2
    size_gb=${3:-30}
    volume_name=${4:-nova_nanobody_cache}
    [[ "$volume_offer_id" =~ ^[0-9]+$ && "$size_gb" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
    [[ "$volume_name" =~ ^[A-Za-z0-9_]{1,64}$ ]] || {
      printf 'Volume name must be 1-64 alphanumeric/underscore characters\n' >&2
      exit 2
    }
    run_billable vastai create volume "$volume_offer_id" --size "$size_gb" --name "$volume_name" --raw
    ;;

  offers)
    [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
    volume_id=$2
    extra_query=${3:-'num_gpus=8 gpu_name=RTX_4090 reliability>0.95 disk_space>=80 rentable=true'}
    [[ "$volume_id" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
    volume=$(volume_json "$volume_id") || {
      printf 'Volume %s does not exist\n' "$volume_id" >&2
      exit 1
    }
    machine_id=$(jq -er '.machine_id' <<<"$volume")
    vastai search offers "machine_id=${machine_id} ${extra_query}" -o 'dph'
    ;;

  launch)
    [[ $# -ge 4 && $# -le 5 ]] || { usage >&2; exit 2; }
    gpu_offer_id=$2
    volume_id=$3
    image_ref=$4
    disk_gb=${5:-80}
    [[ "$gpu_offer_id" =~ ^[0-9]+$ && "$volume_id" =~ ^[0-9]+$ && "$disk_gb" =~ ^[0-9]+$ ]] || {
      usage >&2
      exit 2
    }
    image_tail=${image_ref##*/}
    if [[ "$image_ref" == *@* ]]; then
      image_is_immutable=false
      [[ "$image_ref" =~ ^[^@]+@sha256:[0-9a-f]{64}$ ]] && image_is_immutable=true
    else
      image_is_immutable=false
      [[ "$image_tail" == *:* && "$image_ref" != *:latest ]] && image_is_immutable=true
    fi
    if [[ "$image_is_immutable" != true ]]; then
      printf 'Use an immutable non-latest image tag or digest, not %s\n' "$image_ref" >&2
      exit 2
    fi

    volume=$(volume_json "$volume_id") || {
      printf 'Volume %s does not exist\n' "$volume_id" >&2
      exit 1
    }
    volume_machine=$(jq -er '.machine_id' <<<"$volume")
    gpu_offer=$(offer_json_on_machine "$gpu_offer_id" "$volume_machine") || {
      printf 'GPU offer %s is unavailable on volume host %s\n' \
        "$gpu_offer_id" "$volume_machine" >&2
      exit 1
    }
    offer_machine=$(jq -er '.machine_id' <<<"$gpu_offer")
    if [[ "$volume_machine" != "$offer_machine" ]]; then
      printf 'Volume %s is on machine %s, but offer %s is on machine %s\n' \
        "$volume_id" "$volume_machine" "$gpu_offer_id" "$offer_machine" >&2
      exit 1
    fi

    run_billable vastai create instance "$gpu_offer_id" \
      --image "$image_ref" \
      --disk "$disk_gb" \
      --ssh \
      --direct \
      --cancel-unavail \
      --label nova-nanobody-pixi \
      --link-volume "$volume_id" \
      --mount-path /mnt/nova-cache \
      --onstart "$onstart_script" \
      --raw
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac

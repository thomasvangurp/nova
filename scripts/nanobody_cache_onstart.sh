#!/usr/bin/env bash
set -euo pipefail

cache_root=${NOVA_CACHE_ROOT:-/mnt/nova-cache}

if [[ ! -d "$cache_root" || ! -w "$cache_root" ]]; then
  printf 'Persistent cache is not mounted read/write at %s\n' "$cache_root" >&2
  exit 1
fi

# A Vast local volume is mounted before the on-start script runs. Requiring a
# real mount guards against accidentally placing multi-gigabyte checkpoints in
# disposable container storage when attachment failed.
if ! findmnt -n --target "$cache_root" >/dev/null; then
  printf '%s is a directory, but not a mounted filesystem\n' "$cache_root" >&2
  exit 1
fi

ensure_cache_link() {
  local relative_path=$1
  local destination=$2
  local target="${cache_root}/${relative_path}"

  mkdir -p "$target" "$(dirname "$destination")"

  if [[ -L "$destination" ]]; then
    if [[ "$(readlink "$destination")" != "$target" ]]; then
      printf 'Refusing to replace unexpected symlink %s -> %s\n' \
        "$destination" "$(readlink "$destination")" >&2
      return 1
    fi
    return 0
  fi

  if [[ -e "$destination" ]]; then
    if [[ ! -d "$destination" || -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      printf 'Refusing to replace non-empty cache path %s\n' "$destination" >&2
      return 1
    fi
    rmdir "$destination"
  fi

  ln -s "$target" "$destination"
}

ensure_cache_link boltzgen /root/.boltzgen
ensure_cache_link huggingface /root/.cache/huggingface
ensure_cache_link triton-bgen /root/.cache/triton-bgen
ensure_cache_link cuda-compute /root/.nv/ComputeCache

printf '%s\n' \
  'layout_version=1' \
  'checkpoints=boltzgen' \
  'huggingface_cache=huggingface' \
  'triton_cache=triton-bgen' \
  'cuda_cache=cuda-compute' \
  > "${cache_root}/.nova-nanobody-cache"

printf 'NOVA nanobody cache ready at %s\n' "$cache_root"

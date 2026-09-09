#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 IMAGE_REPOSITORY [VERSION] [--push|--load]" \
    "" \
    "Builds linux/amd64 NOVA nanobody runtime from a clean Git commit." \
    "VERSION defaults to git-<12-character commit>. 'latest' is rejected." \
    "Use --push for Vast; it must be able to pull the published image."
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

image_repository=$1
shift

version=""
output_mode="--load"
for argument in "$@"; do
  case "$argument" in
    --push|--load)
      output_mode=$argument
      ;;
    *)
      if [[ -n "$version" ]]; then
        usage >&2
        exit 2
      fi
      version=$argument
      ;;
  esac
done

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf '%s\n' \
    "Refusing to version an image from a dirty worktree." \
    "Commit the Pixi lockfile, source, and deployment files first." >&2
  exit 1
fi

test -s pixi.toml
test -s pixi.lock
test -s NOVA-nanobody-filter/metanano/__init__.py

vcs_ref=$(git rev-parse HEAD)
short_ref=$(git rev-parse --short=12 HEAD)
created=$(git show -s --format=%cI HEAD)
if command -v sha256sum >/dev/null; then
  lock_sha256=$(sha256sum pixi.lock | awk '{print $1}')
else
  lock_sha256=$(shasum -a 256 pixi.lock | awk '{print $1}')
fi

if [[ -z "$version" ]]; then
  version="git-${short_ref}"
fi
if [[ "$version" == "latest" || ! "$version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'Invalid immutable image version: %s\n' "$version" >&2
  exit 2
fi
if [[ "$image_repository" == *@* || "${image_repository##*/}" == *:* ]]; then
  printf 'IMAGE_REPOSITORY must not contain a tag or digest: %s\n' "$image_repository" >&2
  exit 2
fi

image_ref="${image_repository}:${version}"

docker buildx build \
  --platform linux/amd64 \
  --file docker/nanobody-runtime.Dockerfile \
  --tag "$image_ref" \
  --build-arg "NOVA_RUNTIME_VERSION=${version}" \
  --build-arg "NOVA_VCS_REF=${vcs_ref}" \
  --build-arg "NOVA_CREATED=${created}" \
  --build-arg "PIXI_LOCK_SHA256=${lock_sha256}" \
  --provenance=true \
  --sbom=true \
  "$output_mode" \
  .

printf 'Built %s\n' "$image_ref"
if [[ "$output_mode" == "--push" ]]; then
  digest=$(docker buildx imagetools inspect "$image_ref" --format '{{json .Manifest.Digest}}' | tr -d '"')
  printf 'Deploy by digest: %s@%s\n' "$image_repository" "$digest"
else
  printf '%s\n' "Push this immutable tag before using it on Vast."
fi

# Fast nanobody deployments

The repeat-deployment path has two independent accelerators:

1. A versioned OCI image contains both environments installed from the
   committed `pixi.lock`, the pinned Pixi binary, NOVA source, MetaNano source,
   and the Pixi-provided IgBLAST executable.
2. A Vast local volume holds BoltzGen checkpoints and Hugging Face, Triton, and
   CUDA caches. The volume survives instance destruction.

No checkpoints, API keys, SSH keys, or registry credentials are included in
the image. The canonical `ansible/nanobody.yml` remains the source of artifact
checksums and runtime canaries. On the fast path its Pixi installation is a
locked no-op and its checkpoint step verifies files already on the volume.

## Build and publish an immutable image

Build only from a clean, committed worktree. The helper rejects dirty source
and the `latest` tag so an image version always names one Git revision.

```bash
docker login registry.gitlab.com
scripts/build_nanobody_runtime.sh \
  registry.gitlab.com/thomasvangurp/nova-compound-miner/nanobody-runtime \
  git-$(git rev-parse --short=12 HEAD) \
  --push
```

The command prints the registry digest after pushing. Record and deploy the
`repository@sha256:...` reference; the versioned tag is useful for humans, while
the digest prevents tag movement from changing a deployment.

The image is Linux/amd64 and is based on a digest-pinned CUDA 12.6.3 Ubuntu
22.04 image. Pixi 0.73.0 is installed from a checksum-pinned archive. Both
`bgen` and `metanano` are installed with `pixi install --locked`. IgBLAST comes
from the MetaNano environment; the legacy path is only a symlink to that Pixi
binary.

## Create the persistent cache once

Vast volumes are local: a volume can only be attached to another instance on
the same physical machine. Start with a desirable GPU offer, find volume offers
on that exact machine, then rent one. Mutating helper commands are dry runs
unless `NOVA_CONFIRM_SPEND=1` is set.

```bash
# Pick a current eight-GPU offer using the usual reliability/bandwidth checks.
# This view includes the machine ID needed to locate co-resident storage.
scripts/vast_nanobody_fast_deploy.sh gpu-offers
MACHINE_ID=123456

scripts/vast_nanobody_fast_deploy.sh volume-offers "$MACHINE_ID" 30

# Pick a volume offer from that result. The default 30 GiB leaves headroom over
# the approximately 6 GiB checkpoint set and compiler/download caches.
NOVA_CONFIRM_SPEND=1 \
  scripts/vast_nanobody_fast_deploy.sh volume-create 87654321 30 nova_nanobody_cache

scripts/vast_nanobody_fast_deploy.sh volumes
```

Volume storage is billed until the volume is deleted. Do not store unique
results only in a cache volume; fetch scoring outputs before destroying an
instance, and keep durable results in normal project storage.

## Launch on the volume's host

Search only the volume's physical host, select a returned offer, and launch the
prebuilt image by digest:

```bash
VOLUME_ID=11223344
IMAGE_REF='registry.gitlab.com/thomasvangurp/nova-compound-miner/nanobody-runtime@sha256:REPLACE_ME'

scripts/vast_nanobody_fast_deploy.sh offers "$VOLUME_ID"

NOVA_CONFIRM_SPEND=1 \
  scripts/vast_nanobody_fast_deploy.sh launch 12345678 "$VOLUME_ID" "$IMAGE_REF" 80
```

The launch helper verifies the GPU offer and volume have the same `machine_id`,
attaches the volume at `/mnt/nova-cache`, and installs an idempotent on-start
script. The script refuses to continue if the mount is missing or if a
non-empty local cache would be overwritten. It links these paths to the volume:

- `/root/.boltzgen`
- `/root/.cache/huggingface`
- `/root/.cache/triton-bgen`
- `/root/.nv/ComputeCache`

The registry image must be readable by Vast. Publish it without secrets. If a
private registry is required, configure registry credentials in Vast rather
than adding credentials to this repository or image.

## Fast Ansible verification

Once SSH is ready, run the fast wrapper playbook with the same inventory and
variables as the canonical deployment:

```bash
ANSIBLE_CONFIG=ansible/ansible.cfg ansible-playbook \
  -i 'HOST,' ansible/nanobody-fast.yml \
  -e ansible_port=PORT \
  -e expected_runtime_version=git-$(git rev-parse --short=12 HEAD) \
  --private-key SSH_KEY
```

The preflight requires a clean controller checkout and fails before remote
mutation if the image's embedded Git revision or lock digest differs from that
checkout, either Pixi environment is missing, the volume is not mounted, or the
cache links are wrong. It then runs `nanobody.yml` to apply configuration and
execute its artifact/runtime canaries.

## Fallbacks

- If no eight-GPU offer is available on the volume's host, launch the OCI image
  on another host without the volume and run `nanobody.yml`; only checkpoints
  and cold compiler caches need downloading.
- If the OCI image is unavailable, launch the pinned CUDA base image and run
  `nanobody.yml`; this is the slower full rebuild path and remains authoritative.
- A local Vast volume cannot migrate to another host. Create a second volume
  and use `vastai copy V.OLD:/ V.NEW:/` when host migration is worth the cost.

Destroying an instance preserves an attached volume. Deleting a volume is a
separate irreversible operation and is intentionally not automated here.

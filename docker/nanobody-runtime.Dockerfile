# syntax=docker/dockerfile:1.7

# Keep the platform image immutable. The tag documents the CUDA/userspace ABI;
# the digest prevents an upstream tag update from changing a released runtime.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1

ARG NOVA_RUNTIME_VERSION
ARG NOVA_VCS_REF
ARG NOVA_CREATED
ARG PIXI_LOCK_SHA256

LABEL org.opencontainers.image.title="NOVA nanobody runtime" \
      org.opencontainers.image.description="Locked BoltzGen and MetaNano Pixi environments for NOVA nanobody scoring" \
      org.opencontainers.image.source="https://gitlab.com/thomasvangurp/nova-compound-miner" \
      org.opencontainers.image.version="${NOVA_RUNTIME_VERSION}" \
      org.opencontainers.image.revision="${NOVA_VCS_REF}" \
      org.opencontainers.image.created="${NOVA_CREATED}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIXI_CACHE_DIR=/root/.cache/rattler \
    NOVA_RUNTIME_IMAGE=1

WORKDIR /root

# Pixi is checksum-pinned to the same release used by Ansible. Keeping the
# archive at Ansible's cache path makes its get_url task a checksum-only no-op.
ADD --checksum=sha256:682564e7cd35d38307df3cd6c8e9a45a1437d20febac0c8d165809d7c38408d1 \
    https://github.com/prefix-dev/pixi/releases/download/v0.73.0/pixi-x86_64-unknown-linux-musl.tar.gz \
    /root/.cache/pixi-0.73.0-linux.tar.gz
RUN tar -xzf /root/.cache/pixi-0.73.0-linux.tar.gz -C /usr/local/bin pixi \
    && test "$(pixi --version)" = "pixi 0.73.0"

# Copy only reviewed runtime inputs. Checkpoints and downloaded model caches are
# deliberately absent; Vast mounts them from persistent storage at runtime.
COPY pixi.toml pixi.lock /root/
COPY external_tools/boltzgen/ /root/external_tools/boltzgen/
COPY NOVA-nanobody-filter/ /root/external_tools/NOVA-nanobody-filter/
COPY nanobody_opt/ /root/nanobody_opt/
COPY config/ /root/config/
COPY utils/ /root/utils/
COPY data/igblast_database/database/ /root/external_tools/igblast/database/
COPY data/igblast_internal_data/camelid/ /root/external_tools/igblast/internal_data/camelid/
COPY ansible/files/root/bgen_shim/ /root/bgen_shim/
COPY ansible/files/metanano/ /root/metanano_shim/
COPY ansible/files/root/metanano_shim/ /root/metanano_shim/
COPY --chmod=0755 ansible/files/root/bgen_python.sh /root/bgen_python.sh
COPY --chmod=0755 ansible/files/root/metanano_python.sh /root/metanano_python.sh

RUN test -n "${NOVA_RUNTIME_VERSION}" \
    && test -n "${NOVA_VCS_REF}" \
    && test -n "${NOVA_CREATED}" \
    && test -n "${PIXI_LOCK_SHA256}" \
    && echo "${PIXI_LOCK_SHA256}  /root/pixi.lock" | sha256sum --check --strict

# BuildKit caches accelerate rebuilds without placing solver/download caches in
# the final image. Runtime packages still come exclusively from pixi.lock.
RUN --mount=type=cache,id=nova-nanobody-pixi,target=/root/.cache/rattler,sharing=locked \
    pixi install --locked --environment bgen \
    && pixi install --locked --environment metanano

# The validator's IgBLAST adapter has a stable legacy binary path. The binary is
# supplied by the locked MetaNano Pixi environment; only a compatibility link is
# created here (there is no out-of-band BLAST installation). Image builds
# normally have no GPU, so the runtime Ansible pass executes the GPU/Triton
# canary after launch while this layer validates imports only.
RUN mkdir -p /root/external_tools/igblast/bin \
    && ln -s /root/.pixi/envs/metanano/bin/igblastp /root/external_tools/igblast/bin/igblastp \
    && test -x /root/external_tools/igblast/bin/igblastp \
    && test -x /root/.pixi/envs/metanano/bin/TNP \
    && test -x /root/.pixi/envs/metanano/bin/mkdssp \
    && pixi run --environment bgen python -c "import torch,triton; print(torch.__version__)" \
    && /root/bgen_python.sh -c "import boltzgen" \
    && /root/metanano_python.sh -c "import Bio, torch; print(Bio.__version__)"

RUN printf '%s\n' \
      "runtime_version=${NOVA_RUNTIME_VERSION}" \
      "vcs_ref=${NOVA_VCS_REF}" \
      "created=${NOVA_CREATED}" \
      "pixi_version=0.73.0" \
      "pixi_lock_sha256=${PIXI_LOCK_SHA256}" \
      > /etc/nova-nanobody-runtime

CMD ["bash"]

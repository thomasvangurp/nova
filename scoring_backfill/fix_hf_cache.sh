#!/bin/bash
# Pre-populate the HF cache for boltzgen-1 checkpoints by symlinking the
# already-staged /root/.boltzgen/*.ckpt into the huggingface_hub cache layout,
# so the scorer's configure step does not re-download ~5.6 GB unauthenticated.
set -e
REV=c1be29e1f82ffcc72264f64b993c43fb4e0d17f0
BASE=/root/.cache/huggingface/hub/models--boltzgen--boltzgen-1
mkdir -p $BASE/blobs $BASE/snapshots/$REV $BASE/refs
echo $REV > $BASE/refs/main
declare -A CKPTS=(
  [boltzgen1_diverse.ckpt]=360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c
  [boltzgen1_adherence.ckpt]=ac7078b3dc13064c68e0c3fd542e5bc538c33558bf6607f65e499eb336ca5e5d
  [boltzgen1_ifold.ckpt]=dd4cf108c94471bdc3a326b7b180fa3854dc019110fae780208c30b50bd56578
  [boltz2_conf_final.ckpt]=525a51ef306da7282a54d23a4a5b91212fc60d0ff6b23b56dd6351de3b387530
)
for name in "${!CKPTS[@]}"; do
  sha=${CKPTS[$name]}
  test -f /root/.boltzgen/$name || { echo "MISSING /root/.boltzgen/$name"; exit 1; }
  rm -f $BASE/blobs/$sha
  ln -sf /root/.boltzgen/$name $BASE/blobs/$sha
  rm -f $BASE/snapshots/$REV/$name
  ln -sf ../../blobs/$sha $BASE/snapshots/$REV/$name
done
echo "HF_CACHE_SYMLINKED"

#!/usr/bin/env bash
# The one way to deploy the nanobody miner. Run it from the repository root.
#
# Every argument below has cost an epoch at least once by being forgotten or
# mistyped on an interactive command line: a wrong -i path resolves to an empty
# inventory and the run "succeeds" against nothing; a missing coldkey ss58
# stops the play at the fleet-discovery assertion; a missing token file leaves
# the uploader unauthenticated. They live here so no one reconstructs them.
#
# Nothing secret is in this file. The coldkey ss58 is a public address and
# NOVA_GITHUB_TOKEN_FILE is a path -- Ansible reads that file itself under
# no_log and the value reaches no command line.
set -euo pipefail

# Run from ansible/: ansible.cfg lives there and every relative path below
# -- the inventory, the profile -- is written relative to it.
cd "$(dirname "${BASH_SOURCE[0]}")/../ansible"

INVENTORY=${NOVA_INVENTORY:-inventory/vast_nanobody.py}
COLDKEY=${NOVA_COLDKEY_SS58:-5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ}
TOKEN_FILE=${NOVA_GITHUB_TOKEN_FILE:-$HOME/.nova-github-token}
PROFILE=${NOVA_PROFILE:-vars/nanobody-high-throughput.yml}
PLAYBOOK=${1:-nanobody_services.yml}

# The miner is the ONLY writer of /root/artifacts/chain_state.json, which the
# publisher reads for the block, the epoch and the neuron rows it discovers the
# fleet from. Deploying without these two arguments stops the miner and leaves
# the publisher throwing "miner chain-state is stale" every cycle while looking
# alive -- which is exactly what one deploy without them did. Set
# NOVA_CHAIN_WRITER=false only when deliberately deploying code with no signer.
CHAIN_WRITER=${NOVA_CHAIN_WRITER:-true}

# The hotkey the miner signs with must be one we still hold. Hardcoding a name
# is the stale-configuration failure this repo exists to avoid: nova68-slot-1
# was deregistered and the miner then failed publish_metadata_extrinsic with
# "Custom error: 19" every cycle. Resolved from the chain; override only
# deliberately.
CHAIN_WRITER_HOTKEY="${NOVA_CHAIN_WRITER_HOTKEY:-}"
if [ -z "$CHAIN_WRITER_HOTKEY" ]; then
  CHAIN_WRITER_HOTKEY=$(../deploy/resolve_chain_writer.py "$INVENTORY" "$COLDKEY") || true
fi
if [ -z "$CHAIN_WRITER_HOTKEY" ]; then
  echo "no registered hotkey found for the chain writer; set NOVA_CHAIN_WRITER_HOTKEY" >&2
  exit 2
fi
echo "chain writer signs with: $CHAIN_WRITER_HOTKEY"

# The producer is the service the GPU rental exists to run, and the playbook
# gates it off by default. Omitting this is a green deploy that leaves nothing
# generating candidates.
PRODUCER=${NOVA_PRODUCER:-true}

[ -x "$INVENTORY" ] || { echo "inventory not executable: $INVENTORY" >&2; exit 2; }
[ -r "$TOKEN_FILE" ] || { echo "github token file not readable: $TOKEN_FILE" >&2; exit 2; }

: "${SN68_RELEASE_DIR:?Set SN68_RELEASE_DIR to the downloaded passing CI artifact}"
: "${SN68_TESTED_SHA:?Set SN68_TESTED_SHA to the successful CI commit}"
python3 ../scripts/release_manifest.py verify --root .. \
  --manifest "$SN68_RELEASE_DIR/build_stamp.json" --expected-sha "$SN68_TESTED_SHA"
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  -e "sn68_release_archive=$SN68_RELEASE_DIR/sn68-release.tar.gz" \
  -e "sn68_release_manifest=$SN68_RELEASE_DIR/build_stamp.json" \
  -e "sn68_tested_sha=$SN68_TESTED_SHA" \
  -e "@$PROFILE" \
  -e "nanobody_coldkey_ss58=$COLDKEY" \
  -e "nanobody_github_token_file=$TOKEN_FILE" \
  -e "nanobody_chain_writer_activate=$CHAIN_WRITER" \
  -e "nanobody_chain_writer_hotkey=$CHAIN_WRITER_HOTKEY" \
  -e "nanobody_producer_activate=$PRODUCER" \
  "${@:2}"

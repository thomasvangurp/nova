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

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INVENTORY=${NOVA_INVENTORY:-ansible/inventory/vast_nanobody.py}
COLDKEY=${NOVA_COLDKEY_SS58:-5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ}
TOKEN_FILE=${NOVA_GITHUB_TOKEN_FILE:-$HOME/.nova-github-token}
PROFILE=${NOVA_PROFILE:-vars/nanobody-high-throughput.yml}
PLAYBOOK=${1:-ansible/nanobody_services.yml}

[ -x "$INVENTORY" ] || { echo "inventory not executable: $INVENTORY" >&2; exit 2; }
[ -r "$TOKEN_FILE" ] || { echo "github token file not readable: $TOKEN_FILE" >&2; exit 2; }

exec ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  -e "@$PROFILE" \
  -e "nanobody_coldkey_ss58=$COLDKEY" \
  -e "nanobody_github_token_file=$TOKEN_FILE" \
  "${@:2}"

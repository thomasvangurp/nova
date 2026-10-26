import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import subprocess
import time
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError
from async_substrate_interface.sync_substrate import SubstrateInterface
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "utils"))

# Bittensor 10 exports classes in PascalCase.  Keep the miner compatible with
# the older example API without forcing the scoring environment onto an old
# SDK release.
BTSubtensor = getattr(bt, "Subtensor", getattr(bt, "subtensor", None))
BTAsyncSubtensor = getattr(bt, "AsyncSubtensor", getattr(bt, "async_subtensor", None))
BTWallet = getattr(bt, "Wallet", getattr(bt, "wallet", None))
BTConfig = getattr(bt, "Config", getattr(bt, "config", None))

from config.config_loader import load_config
from nanobody_opt import chain_state
from constants import ALLOWED_AAS
from files import upload_file_to_github
from btdr import QuicknetBittensorDrandTimelock
from combinatorial_db.reactions import get_random_reaction_product
from compound_miner.provider import from_challenge


def _publish_chain_state(state: Dict[str, Any], block: int) -> None:
    """Expose the miner's existing chain view without another WebSocket client.

    The payload shape and the atomic write live in ``nanobody_opt.chain_state``,
    which imports with only the standard library -- the publisher runs in a
    different Pixi environment and the tests in a third, and a contract only one
    of them can import is not a contract.
    """
    path_value = os.environ.get("NOVA_CHAIN_STATE")
    if not path_value or state.get("last_chain_state_block") == int(block):
        return
    metagraph = state.get("metagraph")
    from nanobody_opt import rules
    epoch = int(block) // int(state["epoch_length"])
    epoch_challenge = None
    target = state["config"].nanobody_target
    if epoch >= rules.ACTIVATION_EPOCH:
        if state.get("last_challenge_epoch") != epoch:
            return
        profile = rules.pinned_config(epoch)
        epoch_challenge = rules.challenge(profile, epoch, state.get("epoch_start_block_hash"),
                                          int(state["epoch_length"]))
        target = epoch_challenge["target"]
    try:
        payload = chain_state.build_payload(
            netuid=int(state["config"].netuid),
            block=int(block),
            epoch_length=int(state["epoch_length"]),
            nanobody_target=target,
            epoch_challenge=epoch_challenge,
            miner_uid=int(state["miner_uid"]),
            # The block the NEURON ROWS were synced at, not the block being
            # written. Without it build_payload falls back to `block`, so the
            # roster age read as 0 on every write and a frozen roster was
            # indistinguishable from a fresh one.
            roster_block=state.get("roster_block"),
            neurons=chain_state.neuron_rows(
                getattr(metagraph, "hotkeys", None),
                getattr(metagraph, "coldkeys", None),
            ),
            now=time.time(),
        )
    except chain_state.EmptyRosterError as error:
        # Leave the last good file in place: an empty roster is indistinguishable
        # from a real deregistration once the publisher reads it.
        bt.logging.warning(f"Chain state not published: {error}")
        return
    chain_state.write_atomic(Path(path_value), payload)
    state["last_chain_state_block"] = int(block)


def get_challenge_params_from_blockhash(
    block_hash: str,
    small_molecule_target: str,
    nanobody_target: str,
    include_reaction: bool = False,
) -> Dict[str, Any]:
    """Derive the mechanism-0 challenge without importing optional nano tools."""
    if not isinstance(block_hash, str) or not block_hash.startswith("0x"):
        raise ValueError("block_hash must start with '0x'")
    seed = int(block_hash[2:], 16)
    result: Dict[str, Any] = {
        "small_molecule_target": small_molecule_target,
        "nanobody_target": nanobody_target,
    }
    if include_reaction:
        result["allowed_reaction"] = f"rxn:{seed % 5 + 1}"
    return result


def bind_nanobody_challenge(params, epoch, block_hash, epoch_length):
    from nanobody_opt import rules
    result = {**params, "epoch": epoch, "epoch_start_block_hash": block_hash}
    if epoch >= rules.ACTIVATION_EPOCH:
        context = rules.challenge(rules.pinned_config(epoch), epoch, block_hash, epoch_length)
        result.update(nanobody_target=context["target"], allowed_v=context["allowed_v"],
                      allowed_cdr3_length=context["allowed_cdr3_length"])
    return result

'''
This example miner contains all logic necessary for submitting a response to the chain
in a format that will be accepted by the validator.

Each miner should implement their own logic for selecting candidate molecules and nanobodies.
The placeholder selection logic here is a random selection of a small molecule from the combinatorial db 
and a random mutation of a base nanobody sequence (therefore not competitive).
'''

# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    # Bittensor 10 defaults to ignoring process arguments for library callers.
    # A standalone miner must opt back into its CLI contract.
    os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=379, help="The chain subnet uid.") #68 for mainnet
    parser.add_argument(
        '--chain-state-only',
        action='store_true',
        default=os.getenv('NOVA_CHAIN_STATE_ONLY') == '1',
        help=(
            "publish chain state but never commit candidates. Use when another "
            "process owns submission: this miner's candidates are the upstream "
            "example (a random mutation of a base sequence), and committing "
            "them overwrites the real submission on the hotkey it signs with"
        ),
    )
    # Bittensor standard argument additions.
    BTSubtensor.add_args(parser)
    bt.logging.add_args(parser)
    BTWallet.add_args(parser)

    # Parse combined config
    config = BTConfig(parser, args=sys.argv[1:])

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


# Moved to utils/files.py, which the commit driver already imports; a second
# copy here would drift from the one that writes the on-chain content.
from files import load_github_path  # noqa: E402  (re-exported for callers)


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

def _resolved_chain_endpoint(config: argparse.Namespace) -> str:
    """Return the websocket URI behind a Bittensor network alias.

    async-substrate-interface 2.x wants a real URI, while the Bittensor CLI
    accepts aliases such as ``finney`` and resolves them under
    ``config.subtensor.chain_endpoint``.  Passing the alias straight through
    fails only when the first query runs, long after startup looked healthy.
    """
    chain_endpoint = config.subtensor.chain_endpoint
    if not str(chain_endpoint).startswith(("ws://", "wss://")):
        raise ValueError(f"Invalid resolved chain endpoint: {chain_endpoint!r}")
    return str(chain_endpoint)


async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = BTWallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        subtensor = BTAsyncSubtensor(network=config.network)
        await subtensor.initialize()
        bt.logging.info(f"Connected to subtensor network: {config.network}")

        # Sync metagraph while keeping the underlying websocket alive for the
        # submission loop.  Returning from an ``async with`` block here closes
        # the connection before the first commitment is sent.
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        bt.logging.info(f"Metagraph synced successfully.")

        bt.logging.info(f"Subtensor: {subtensor}")
        bt.logging.info(f"Metagraph synced: {metagraph}")

        # Get miner UID
        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Miner UID: {miner_uid}")

        # Query epoch length
        node = SubstrateInterface(url=_resolved_chain_endpoint(config))
        # Set epoch_length to tempo + 1
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. ADD YOUR OWN CODE FOR MINING
# ----------------------------------------------------------------------------

def get_random_small_molecule(
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """
    Uses combinatorial_db.reactions to pick a random reaction and valid
    reactant IDs, then returns the product name in the format
    'rxn:rxn_id:mol1_id:mol2_id' or 'rxn:rxn_id:mol1_id:mol2_id:mol3_id'.
    Returns None if the DB is missing or no valid combination is found.
    """
    return get_random_reaction_product(rng=rng)


def mutate_sequence(
    sequence: str,
    num_mutations: int,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Mutates a variable number of positions in a sequence by replacing residues
    with one of the allowed amino acids from utils.constants.ALLOWED_AAS.

    Args:
        sequence: The base amino acid sequence (e.g. nanobody one-letter codes).
        num_mutations: Number of positions to mutate. Clamped to [0, len(sequence)].
        rng: Optional random.Random instance for reproducible randomness.

    Returns:
        A new string with exactly num_mutations positions changed to random
        allowed AAs (always to a different residue at each chosen position).
    """
    if not sequence or num_mutations <= 0:
        return sequence
    rng = rng or random
    allowed = list(ALLOWED_AAS)
    seq_list = list(sequence)
    length = len(seq_list)
    num_mutations = min(num_mutations, length)
    indices = rng.sample(range(length), num_mutations)
    for i in indices:
        current = seq_list[i]
        choices = [a for a in allowed if a != current]
        if not choices:
            continue
        seq_list[i] = rng.choice(choices)
    return "".join(seq_list)


def dummy_get_candidate_products(challenge_params: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Returns a dummy candidate small molecule and nanobody.
    If n>1, items should be separated by ',' e.g. 'rxn:3:49485:2099:70633,rxn:3:49485:2099:70634'
    """
    try:
        if challenge_params is None:
            raise ValueError("challenge parameters are required for competitive molecule selection")
        candidate_small_molecules = from_challenge(challenge_params)
    except Exception as e:
        bt.logging.error(f"Error getting candidate small molecule: {e}")
        candidate_small_molecules = "~"

    try:
        base_nanobody = 'EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS'
        candidate_nanobodies = mutate_sequence(base_nanobody, num_mutations=random.randint(1, 30))
    except Exception as e:
        bt.logging.error(f"Error getting candidate nanobody: {e}")
        candidate_nanobodies = "~"
    
    return candidate_small_molecules, candidate_nanobodies


async def schedule_compound_search(
    state: Dict[str, Any], challenge_params: Dict[str, Any], blocks_remaining: int
) -> None:
    """Start the challenge-bound search policy without blocking commitments."""
    if os.environ.get("NOVA_LIVE_SEARCH", "0") != "1":
        return
    if blocks_remaining <= 25:
        bt.logging.info(
            f"Skipping compound search with only {blocks_remaining} blocks remaining"
        )
        return
    previous = state.get("compound_search_process")
    if previous is not None and previous.returncode is None:
        previous.terminate()
        try:
            await asyncio.wait_for(previous.wait(), timeout=15)
        except asyncio.TimeoutError:
            previous.kill()
            await previous.wait()

    targets = challenge_params.get("small_molecule_target") or []
    if isinstance(targets, str):
        targets = [targets]
    if len(targets) != 1:
        bt.logging.error(f"Cannot schedule compound search for targets={targets!r}")
        return
    reaction = int(str(challenge_params["allowed_reaction"]).split(":", 1)[1])
    epoch = int(challenge_params["epoch"])
    target = str(targets[0])
    result_template = os.environ.get(
        "NOVA_CANDIDATE_RESULT", "artifacts/result_rxn{reaction}.json"
    )
    result = result_template.format(reaction=reaction)
    search_python = os.environ.get("NOVA_SEARCH_PYTHON", sys.executable)
    deadline_seconds = max(60, (blocks_remaining - 15) * 12)
    log_path = Path(BASE_DIR) / "artifacts" / "live" / "search" / (
        f"{target}-epoch{epoch}-rxn{reaction}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        search_python,
        str(Path(BASE_DIR) / "tools" / "run_epoch_search.py"),
        "--target",
        target,
        "--epoch",
        str(epoch),
        "--reaction",
        str(reaction),
        "--result",
        result,
        "--deadline-seconds",
        str(deadline_seconds),
        "--gpus",
        os.environ.get("NOVA_SEARCH_GPUS", "0,1"),
        "--shards-per-gpu",
        os.environ.get("NOVA_SEARCH_SHARDS_PER_GPU", "3"),
        "--construction-workers",
        os.environ.get("NOVA_SEARCH_CONSTRUCTION_WORKERS", "40"),
        "--similarity-workers",
        os.environ.get("NOVA_SEARCH_SIMILARITY_WORKERS", "32"),
        "--proposals",
        os.environ.get("NOVA_SEARCH_PROPOSALS", "250000"),
        "--reduced-limit",
        os.environ.get("NOVA_SEARCH_REDUCED_LIMIT", "1024"),
        "--exact-limit",
        os.environ.get("NOVA_SEARCH_EXACT_LIMIT", "128"),
    ]
    if os.environ.get("NOVA_SEARCH_DIVERSE_RETENTION", "0") == "1":
        arguments.append("--diverse-retention")
    with log_path.open("ab", buffering=0) as log:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=BASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["compound_search_process"] = process
    bt.logging.info(
        f"Started compound search pid={process.pid} target={target} "
        f"epoch={epoch} reaction={reaction} deadline_seconds={deadline_seconds}"
    )


async def initial_epoch_schedule(
    subtensor: Any, epoch_length: int
) -> Tuple[int, int, int]:
    """Resolve the initial challenge boundary and its live search budget.

    A miner starting just before an epoch boundary waits for the fresh epoch.
    The chain height must then be read again: carrying the pre-sleep remaining
    count into ``schedule_compound_search`` shortens a full epoch search to the
    stale tail of the prior epoch.

    Returns:
        ``(current_block, challenge_boundary, blocks_remaining)``.
    """
    current_block = await subtensor.get_current_block()
    next_boundary = ((current_block // epoch_length) + 1) * epoch_length
    blocks_remaining = next_boundary - current_block

    if blocks_remaining < 20:
        bt.logging.info("Too close to epoch end, waiting for next epoch to start...")
        await asyncio.sleep(12 * blocks_remaining)
        current_block = await subtensor.get_current_block()
        while current_block < next_boundary:
            await asyncio.sleep(min(12, max(1, 12 * (next_boundary - current_block))))
            current_block = await subtensor.get_current_block()

    challenge_boundary = (current_block // epoch_length) * epoch_length
    blocks_remaining = epoch_length - (current_block % epoch_length)
    return current_block, challenge_boundary, blocks_remaining

# ----------------------------------------------------------------------------
# 5. RESPONSE SUBMISSION LOGIC
# ----------------------------------------------------------------------------

async def submit_response(state: Dict[str, Any]) -> bool:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_small_molecules = state['candidate_small_molecules']
    candidate_nanobodies = state['candidate_nanobodies']
    if not candidate_small_molecules and not candidate_nanobodies:
        bt.logging.warning("No candidate small molecules or nanobodies to submit")
        return False

    bt.logging.info(f"Starting submission process for small molecules: {candidate_small_molecules} and nanobodies: {candidate_nanobodies}")
    
    # 1) Encrypt the response

    message = f"{candidate_small_molecules}|{candidate_nanobodies}"
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], message, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        # Read, base64-encode
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        # Generate short hash-based filename
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

    # 3) Attempt chain commitment
    bt.logging.info(f"Attempting chain commitment...")
    try: 
        commitment_status = await state['subtensor'].set_commitment(
            wallet=state['wallet'],
            netuid=state['config'].netuid,
            data=commit_content
        )
        bt.logging.info(f"Chain commitment status: {commitment_status}")
    except MetadataError:
        bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
        return False

    # 4) If chain commitment success, upload to GitHub
    if commitment_status:
        try:
            bt.logging.info(f"Commitment set successfully for {commit_content}")
            bt.logging.info("Attempting GitHub upload...")
            github_status = upload_file_to_github(filename, encoded_content)
            if github_status:
                bt.logging.info(f"File uploaded successfully to {commit_content}")
            else:
                bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
        except Exception as e:
            bt.logging.error(f"Failed to upload file for {commit_content}: {e}")
        # The chain rate limit has been consumed even if the subsequent upload
        # failed, so callers must not immediately retry the same checkpoint.
        return True
    return False


def _checkpoint_changed(state: Dict[str, Any], candidate_small_molecules: str) -> bool:
    """Return whether a molecule checkpoint deserves another chain slot.

    Search publishes progressively better exact checkpoints during an epoch.
    Recommitting an unchanged molecule list can consume the pallet's cooldown
    immediately before a better checkpoint appears.  The placeholder ``~`` is
    exempt because nanobody-only updates still need to remain possible.
    """
    return (
        candidate_small_molecules == "~"
        or candidate_small_molecules != state.get("last_committed_small_molecules")
    )


async def _submit_current_candidates(state: Dict[str, Any]) -> bool:
    """Submit changed candidates and update cooldown state only on chain success.

    Refuses outright in chain-state-only mode.  This process is run here for one
    reason -- it is the only writer of /root/artifacts/chain_state.json, which
    the publisher reads for the block, the epoch and the fleet it discovers from
    the metagraph.  Its CANDIDATES, though, come from the upstream example
    implementation: dummy_get_candidate_products returns a random 1-30 mutation
    of a hardcoded base sequence, which its own docstring calls "therefore not
    competitive".

    Left submitting, it commits that dummy on whichever hotkey it signs with --
    and resolve_chain_writer.py deliberately picks a REGISTERED one, because the
    miner dies otherwise.  So the chain writer is always one of our fleet uids,
    and it overwrites that uid's real submission with noise: observed live on
    uid 19, "Commitment set successfully", racing the publisher's own
    set_commitment for the same uid.
    """
    if state.get("chain_state_only"):
        bt.logging.debug(
            "chain-state-only: not committing example candidates; this process "
            "publishes chain state and nothing else"
        )
        return False
    candidate_small_molecules = state["candidate_small_molecules"]
    if not _checkpoint_changed(state, candidate_small_molecules):
        bt.logging.debug(
            "Exact molecule checkpoint is unchanged; preserving the next "
            "commitment slot for a better anytime checkpoint."
        )
        return False
    committed = await submit_response(state)
    if committed:
        state["last_submitted_block"] = await state["subtensor"].get_current_block()
        state["last_committed_small_molecules"] = candidate_small_molecules
    return committed


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'submission_interval': 1200,
        # When set, publish chain state and never commit: another process owns
        # submission on this hotkey.  See _submit_current_candidates.
        'chain_state_only': bool(getattr(config, 'chain_state_only', False)),

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Bittensor Drand Timelock objects
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_small_molecules': None,
        'candidate_nanobodies': None,
        'last_submitted_block': 0,
        'last_committed_small_molecules': None,
    }

    # setup_bittensor_objects synced the metagraph moments ago; the roster is
    # as fresh as the chain head now.  Record it so the loop below re-syncs on
    # the next EPOCH NUMBER change rather than on an exact block match.
    startup_block = int(await subtensor.get_current_block())
    state["roster_block"] = startup_block
    state["last_epoch_synced"] = startup_block // epoch_length

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block, block_to_check, blocks_remaining = await initial_epoch_schedule(
        subtensor, epoch_length
    )

    block_hash = await subtensor.determine_block_hash(block_to_check)
    challenge_params = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        small_molecule_target=config.small_molecule_target,
        nanobody_target=config.nanobody_target,
        include_reaction=config.random_valid_reaction
    )
    challenge_params = bind_nanobody_challenge(challenge_params, block_to_check // epoch_length,
                                               block_hash, epoch_length)
    # The challenge for this epoch is now in hand; the loop below refreshes it
    # only when the EPOCH NUMBER moves past this one.
    state["last_challenge_epoch"] = challenge_params["epoch"]
    state["epoch_start_block_hash"] = block_hash
    bt.logging.info(f"Challenge params: {challenge_params}")
    _publish_chain_state(state, current_block)
    # A hot code reload may leave the already-running epoch search in its own
    # process group.  Skip only this initial launch so the replacement miner
    # can adopt that search without starting a duplicate; boundary launches
    # below remain enabled.
    if os.environ.get("NOVA_SKIP_INITIAL_SEARCH", "0") != "1":
        await schedule_compound_search(
            state,
            challenge_params,
            blocks_remaining=blocks_remaining,
        )

    # Update state with your results and other relevant info here
    candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products(challenge_params)
    state['candidate_small_molecules'] = candidate_small_molecules
    state['candidate_nanobodies'] = candidate_nanobodies
    # Submit response
    await _submit_current_candidates(state)
    await asyncio.sleep(1)

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # Re-sync whenever the EPOCH NUMBER changes, not when the block is
            # exactly divisible by the epoch length.  The exact-match branch
            # below can be stepped over (a slow chain round-trip, a 3 s error
            # back-off), and when it was the only place the metagraph synced,
            # the roster froze at process start: on 2026-09-16 and again on
            # 2026-10-09 the publisher kept assigning sequences to a uid that
            # had been re-registered to another coldkey.  An inequality on the
            # epoch number fires on the first poll after ANY boundary, however
            # many blocks the loop skipped.  A failed sync leaves
            # last_epoch_synced untouched, so it is retried next iteration.
            epoch_now = current_block // epoch_length
            if state.get("last_epoch_synced") != epoch_now:
                try:
                    await metagraph.sync()
                except Exception as error:  # a failed sync must not end the epoch
                    bt.logging.warning(
                        f"Metagraph epoch re-sync failed at block {current_block}: {error}"
                    )
                else:
                    state["last_epoch_synced"] = epoch_now
                    state["roster_block"] = int(current_block)
                    state["last_chain_state_block"] = None
                    bt.logging.info(
                        f"Metagraph re-synced for epoch {epoch_now} at block {current_block}"
                    )

            _publish_chain_state(state, current_block)

            # New epoch: fetch the new challenge and (re)start the search.
            # Same epoch-number-changed guard as the re-sync above, for the
            # same two reasons.  `current_block % epoch_length == 0` (a) is
            # true for every ~1.3 s poll during the ~12 s the boundary block
            # is the chain head, so this ran ~6 times per boundary -- with
            # NOVA_LIVE_SEARCH=1 that kills and relaunches the compound search
            # each time -- and (b) is never true when a slow round-trip or the
            # 3 s error back-off steps over the boundary block, which left the
            # previous epoch's challenge in place for the whole epoch.  The
            # challenge is always derived from the BOUNDARY block's hash, not
            # the block we happened to observe, so a late detection still
            # reads the right challenge.
            if state.get("last_challenge_epoch") != epoch_now:
                challenge_boundary = epoch_now * epoch_length
                bt.logging.info(
                    f"Found epoch boundary {challenge_boundary} (epoch {epoch_now}) "
                    f"at block {current_block}."
                )

                block_hash = await subtensor.determine_block_hash(challenge_boundary)

                challenge_params = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    small_molecule_target=config.small_molecule_target,
                    nanobody_target=config.nanobody_target,
                    include_reaction=config.random_valid_reaction
                )
                challenge_params = bind_nanobody_challenge(challenge_params, epoch_now,
                                                           block_hash, epoch_length)
                # Mark the epoch handled before anything below can raise: a
                # failure in the search launch or the submit must not re-run
                # this branch (and relaunch the search) on the next poll.  A
                # failure ABOVE (block hash / challenge) leaves it unset, so
                # the challenge fetch is retried next iteration.
                state["last_challenge_epoch"] = epoch_now
                state["epoch_start_block_hash"] = block_hash
                _publish_chain_state(state, current_block)
                await schedule_compound_search(
                    state,
                    challenge_params,
                    blocks_remaining=epoch_length - (current_block % epoch_length),
                )

                # Update state with your results and other relevant info here
                candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products(challenge_params)
                state['candidate_small_molecules'] = candidate_small_molecules
                state['candidate_nanobodies'] = candidate_nanobodies
                # Submit response
                await _submit_current_candidates(state)
                await asyncio.sleep(1)
                
            else:
                # Waiting for epoch
                blocks_remaining = epoch_length - (current_block % epoch_length)
                if (blocks_remaining > 20) and (current_block - state['last_submitted_block'] > 50):
                    bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                    candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products(challenge_params)
                    state['candidate_small_molecules'] = candidate_small_molecules
                    state['candidate_nanobodies'] = candidate_nanobodies
                    await _submit_current_candidates(state)
                    await asyncio.sleep(1)
                await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break

        except asyncio.CancelledError:
            bt.logging.info("Resetting subtensor connection.")
            subtensor = BTAsyncSubtensor(network=config.network)
            await subtensor.initialize()
            await asyncio.sleep(1)
            continue
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

import asyncio
import base64
import os
import random
import requests
import shutil
import subprocess
import tempfile
from dotenv import load_dotenv
import bittensor as bt
from pathlib import Path
from typing import List, Tuple, Dict
import csv

load_dotenv(override=True)

# Defined here rather than in the miner because the commit driver needs it
# too, and it imports `files`, not the miner. Two copies of this would drift
# apart silently: the value goes on-chain, and a validator that cannot fetch
# the payload it points at scores nothing.
def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


def _git_environment(deploy_key: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        f"ssh -i {deploy_key} -o IdentitiesOnly=yes -o BatchMode=yes "
        "-o StrictHostKeyChecking=accept-new"
    )
    return env


def _git(command: List[str], cwd: Path | None, env: Dict[str, str]) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _shared_checkout(remote: str, branch: str, work_root: Path,
                     env: Dict[str, str]) -> Path | None:
    """A checkout reused within an epoch, refreshed to the remote tip.

    `reset --hard` every time is what keeps this equivalent to a fresh clone:
    nothing from a previous uid can survive into the next one, so there is no
    worktree drift for the per-uid clone to have been protecting against.
    Returns None if it cannot be made ready, and the caller clones instead.
    """
    checkout = work_root / "shared"
    try:
        if not (checkout / ".git").is_dir():
            shutil.rmtree(checkout, ignore_errors=True)
            _git(["git", "clone", "--quiet", "--depth", "1", "--single-branch",
                  "--branch", branch, remote, str(checkout)], cwd=None, env=env)
            _git(["git", "config", "user.name", "NOVA miner"], checkout, env)
            _git(["git", "config", "user.email",
                  "nova-miner@users.noreply.github.com"], checkout, env)
            return checkout
        _git(["git", "fetch", "--quiet", "--depth", "1", "origin", branch],
             checkout, env)
        _git(["git", "reset", "--quiet", "--hard", "FETCH_HEAD"], checkout, env)
        _git(["git", "clean", "-qfd"], checkout, env)
        return checkout
    except subprocess.CalledProcessError:
        shutil.rmtree(checkout, ignore_errors=True)
        return None


def _write_and_push(checkout: Path, relative_dir: Path, filename: str,
                    content: bytes, branch: str, env: Dict[str, str]) -> bool:
    destination = checkout / relative_dir / f"{filename}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == content:
        return True
    destination.write_bytes(content)
    relative_file = destination.relative_to(checkout).as_posix()
    _git(["git", "add", "--", relative_file], checkout, env)
    _git(["git", "commit", "--quiet", "-m", f"Encrypted response {filename}"],
         checkout, env)
    _git(["git", "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"],
         checkout, env)
    return True


def _push_via_deploy_key(filename: str, encoded_content: str) -> bool:
    """Publish one encrypted submission with a repository-scoped SSH key.

    Each attempt uses a fresh shallow clone. This avoids persistent-worktree
    drift and makes a non-fast-forward retry safe without resetting or deleting
    a checkout that might contain unrelated state.
    """
    owner = os.environ.get("GITHUB_REPO_OWNER")
    repo = os.environ.get("GITHUB_REPO_NAME")
    branch = os.environ.get("GITHUB_REPO_BRANCH")
    repo_path = os.environ.get("GITHUB_REPO_PATH", "")
    deploy_key = os.environ.get("NOVA_DEPLOY_KEY")
    work_root = Path(os.environ.get("NOVA_GIT_WORKDIR", "/root/.nova_submissions"))
    if not owner or not repo or not branch or not deploy_key:
        raise ValueError("SSH GitHub deployment variables are not set")
    key = Path(deploy_key)
    if not key.is_file():
        raise ValueError(f"GitHub deploy key does not exist: {key}")

    relative_dir = Path(repo_path)
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise ValueError(f"GITHUB_REPO_PATH must stay inside the repository: {repo_path!r}")
    if not filename or filename != Path(filename).name:
        raise ValueError(f"invalid submission filename: {filename!r}")
    content = base64.b64decode(encoded_content, validate=True)
    remote = f"git@github.com:{owner}/{repo}.git"
    env = _git_environment(str(key))
    work_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    # A shared checkout, reused across the uids of one epoch.
    #
    # Cloning per uid is what made a commit cost ~73s, and at ~6 blocks each a
    # twelve-uid fleet cannot finish before the validator stops reading at
    # end-16 -- so the tail committed into the blackout, silently, every epoch.
    # The clone was never about isolating one file from another: its stated
    # purpose is making a non-fast-forward RETRY safe. That is preserved
    # exactly, because any failure here falls through to the fresh-clone path
    # below, which is unchanged.
    shared = _shared_checkout(remote, branch, work_root, env)
    if shared is not None:
        try:
            if _write_and_push(shared, relative_dir, filename, content, branch, env):
                return True
        except subprocess.CalledProcessError:
            # Drift, a racing push, a half-made checkout: throw it away and let
            # the original path do it properly rather than trying to repair it.
            shutil.rmtree(shared, ignore_errors=True)

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(2):
        try:
            with tempfile.TemporaryDirectory(prefix="submission-", dir=work_root) as tmp:
                checkout = Path(tmp) / "repo"
                _git(
                    [
                        "git", "clone", "--quiet", "--depth", "1",
                        "--single-branch", "--branch", branch, remote, str(checkout),
                    ],
                    cwd=None,
                    env=env,
                )
                destination = checkout / relative_dir / f"{filename}.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and destination.read_bytes() == content:
                    return True
                destination.write_bytes(content)
                _git(["git", "config", "user.name", "NOVA miner"], checkout, env)
                _git(
                    ["git", "config", "user.email", "nova-miner@users.noreply.github.com"],
                    checkout,
                    env,
                )
                relative_file = destination.relative_to(checkout).as_posix()
                _git(["git", "add", "--", relative_file], checkout, env)
                _git(
                    ["git", "commit", "--quiet", "-m", f"Encrypted response {filename}"],
                    checkout,
                    env,
                )
                _git(
                    ["git", "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"],
                    checkout,
                    env,
                )
                return True
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == 0:
                continue
    if last_error is not None:
        bt.logging.error(
            "Deploy-key push failed after retry: "
            f"{last_error.stderr.strip() or last_error.stdout.strip()}"
        )
    return False


def upload_file_to_github(filename: str, encoded_content: str):
    if os.environ.get("NOVA_GIT_PUSH", "0") == "1":
        return _push_via_deploy_key(filename, encoded_content)

    # Github configs
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')   # example: nova
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH') # example: main
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER') # example: metanova-labs
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') # example: /data/results or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError("Github environment variables not set. Please set them in your .env file.")

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        }

    # Check if the file already exists (need its SHA to update)
    existing_file = requests.get(url, headers=headers, params={"ref": github_repo_branch})
    sha = existing_file.json().get("sha") if existing_file.status_code == 200 else None

    payload = {
        "message": f"Encrypted response for {filename}",
        "content": encoded_content,
        "branch": github_repo_branch,
    }
    if sha:
        payload["sha"] = sha  # updating existing file

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    else:
        bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
        return False


async def read_local_input_file(file_path, config, subtensor):
    """
    Loads input data from a local file for testing purposes.
    Expected format:
    uid|mol_name1,mol_name2,...|protein_sequence1,protein_sequence2,...
    Multiple lines supported. Creates random block numbers for each entry.
    """
    if not os.path.exists(file_path):
        bt.logging.error(f"Local input file not found: {file_path}")
        return None

    current_block = await subtensor.get_current_block()
    uid_to_data = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        uid, mol_names, protein_sequences = line.strip().split('|')
        uid = int(uid)
        mol_names = mol_names.split(',')
        protein_sequences = protein_sequences.split(',')
        uid_to_data[uid] = {
            'molecules': mol_names,
            'sequences': protein_sequences,
            'block_submitted': current_block - random.randint(11, 360),
            'push_time': ""
        }
    return uid_to_data

def read_fasta(path: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    name = None
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    items.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if name is not None:
            items.append((name, "".join(seq_parts)))
    return items

def write_fasta(items: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, seq in items:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

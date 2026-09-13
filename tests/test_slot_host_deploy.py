"""The purchase host's deployment, pinned where a rebuild would have found it.

Two things ran on the Raspberry Pi for a week without existing in this
repository: the pair of systemd units that push purchased hotkeys to the rented
GPU box, and the Pixi environment the purchase agent's interpreter comes from.
Both were installed by hand. A rebuilt Pi would have come back buying
registrations it never delivered, with a lock file that claimed the environment
contained a vast.ai client it did not have -- and neither failure shows up on the
leaderboard, because a slot whose hotkey never reached the miner looks exactly
like a slot that was never bought.

Every test here reads the real playbook, the real templates and the real lock.
The gate tests evaluate the playbook's OWN expressions in both directions, since
the two defects this deployment keeps producing are a gate that cannot fail and a
gate that fails on a healthy host.

Nothing here imports ``bittensor``, touches the network or reads a credential.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SLOTS = REPOSITORY / "ansible" / "slots.yml"
TEMPLATES = REPOSITORY / "ansible" / "templates"
SLOT_HOST = REPOSITORY / "deploy" / "slot_host"
SYNC_SCRIPT = SLOT_HOST / "sync_hotkeys.sh"
SERVICE_TEMPLATE = TEMPLATES / "nova-hotkey-sync.service.j2"
TIMER_TEMPLATE = TEMPLATES / "nova-hotkey-sync.timer.j2"


# --- reading the deployment as data ---------------------------------------


def _play() -> dict:
    return yaml.safe_load(SLOTS.read_text())[0]


def _tasks() -> list[dict]:
    return _play()["tasks"]


def _task(name: str) -> dict:
    for task in _tasks():
        if task.get("name") == name:
            return task
    raise AssertionError(f"ansible/slots.yml no longer defines the task {name!r}")


def _directives(template: Path) -> str:
    """A unit template with its commentary removed.

    The templates here explain at length which directives they deliberately do
    NOT use, so every check below has to read the effective unit rather than the
    prose around it.
    """
    body = template.read_text()
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )


# Ansible's own regexes are written the Ansible way -- 'N\\s+' as a single
# backslash, which is what collapses whitespace in a rendered message on the
# purchase host and what jinja2's lexer warns about when this suite re-renders
# them here. Doubling the backslash to silence the warning breaks the regex at
# the boundary: the message comes out with its newlines intact, which is how a
# 40-line traceback reaches a deploy recap.
pytestmark = pytest.mark.filterwarnings("ignore:.*invalid escape sequence.*:DeprecationWarning")


def _environment():
    """A Jinja environment with the Ansible filters and tests these gates use.

    The point is to evaluate the strings the playbook actually ships rather than
    a paraphrase of them, so a future edit that weakens a gate fails here.
    """
    import jinja2

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["regex_search"] = lambda value, pattern: (
        match.group(0) if (match := re.search(pattern, value)) else None
    )
    env.filters["regex_replace"] = lambda value, pattern, replacement: re.sub(
        pattern, replacement, value
    )
    env.filters["basename"] = lambda value: value.rsplit("/", 1)[-1]
    env.filters["dirname"] = lambda value: value.rsplit("/", 1)[0]
    env.tests["search"] = lambda value, pattern: re.search(pattern, value) is not None
    env.tests["match"] = lambda value, pattern: re.match(pattern, value) is not None
    return env


def _evaluate(expression: str, **variables) -> bool:
    rendered = _environment().from_string("{{ " + expression + " }}").render(**variables)
    assert rendered in {"True", "False"}, f"{expression!r} rendered {rendered!r}"
    return rendered == "True"


def _render(expression: str, **variables) -> str:
    return _environment().from_string(expression).render(**variables)


# --- 1. the units exist in the repository at all --------------------------


def test_the_hotkey_sync_units_are_deployed_from_this_repository():
    """A hand-installed unit is a unit a rebuild silently drops.

    In production: the Pi held three purchased hotkeys and the box held two,
    because the only thing that moved them was a service and timer that existed
    on exactly one machine and in no playbook.
    """
    assert SERVICE_TEMPLATE.exists(), "the hotkey sync service is not in the repository"
    assert TIMER_TEMPLATE.exists(), "the hotkey sync timer is not in the repository"

    installed = {
        task["ansible.builtin.template"]["src"]: task
        for task in _tasks()
        if "ansible.builtin.template" in task
    }
    for template in ("templates/nova-hotkey-sync.service.j2",
                     "templates/nova-hotkey-sync.timer.j2"):
        assert template in installed, f"ansible/slots.yml never installs {template}"
        task = installed[template]
        assert task["ansible.builtin.template"]["dest"].startswith(
            "/etc/systemd/system/"
        ), f"{template} is not installed as a systemd unit"
        assert task.get("become") is True, f"installing {template} needs become: true"


def test_the_hotkey_sync_is_deployed_on_the_linux_path_only():
    """The macOS path runs the purchaser in tmux and has no systemd at all."""
    offenders = [
        task["name"]
        for task in _tasks()
        if "hotkey sync" in task.get("name", "").lower()
        or "vastai client" in task.get("name", "").lower()
        if "ansible_system == 'Linux'" not in str(task.get("when", ""))
    ]
    assert offenders == [], f"these would also run on macOS, which has no systemd: {offenders}"


def test_every_systemd_action_on_the_purchase_host_is_privileged():
    """A systemctl or journalctl call without become fails as the wallet user."""
    unprivileged = []
    for task in _tasks():
        command = task.get("ansible.builtin.command") or {}
        argv = command.get("argv") if isinstance(command, dict) else None
        first = str(argv[0]) if argv else ""
        systemd_module = "ansible.builtin.systemd_service" in task
        if (first in {"systemctl", "journalctl"} or systemd_module) and not task.get("become"):
            unprivileged.append(task.get("name"))
    assert unprivileged == [], f"these touch systemd without become: true: {unprivileged}"


# --- 2. the unit must not re-derive its configuration --------------------


def test_the_unit_pins_every_value_the_script_would_default():
    """The script's defaults are a second, silent copy of this configuration.

    ``sync_hotkeys.sh`` falls back to wallet ``mining_ops``, prefix
    ``nova68-slot`` and a vastai under ``/opt/nova-nanobody`` when its
    environment says nothing. All three are right today and all three are wrong
    the moment ``slot_manager_wallet``, ``slot_manager_hotkey_prefix`` or
    ``slot_manager_remote_repo`` changes here -- and the prefix is the dangerous
    one, because a prefix that matches nothing makes the script report "no
    hotkeys matching ... yet" and exit 0. This is the boundary version of a
    value that is correct in isolation.
    """
    defaults = set(re.findall(r"\$\{(NOVA_[A-Z_]+):-", SYNC_SCRIPT.read_text()))
    assert defaults, "the sync script no longer takes its configuration from the environment"

    unit = SERVICE_TEMPLATE.read_text()
    for name in sorted(defaults):
        line = re.search(rf"^Environment={name}=(.+)$", unit, re.MULTILINE)
        assert line, (
            f"{name} has a default inside sync_hotkeys.sh but the unit does not set "
            "it, so the deployed service silently uses the script's copy"
        )
        assert "{{" in line.group(1), (
            f"Environment={name} is hardcoded in the unit; it must come from a "
            "playbook variable or it is a third copy of the same value"
        )


def test_the_timer_does_not_fire_the_service_merely_by_starting():
    """``Requires=`` on a timer activates its service the moment the timer starts.

    The hand-installed timer had it, so every ``systemctl restart <timer>`` --
    which is what the deploy does -- fired an unscheduled run before the deploy
    had finished checking that a run could work. The purchase timer next door
    documents the same trap for a unit where the unscheduled run SPENDS TAO.
    """
    timer = _directives(TIMER_TEMPLATE)
    assert "Requires=" not in timer, "a timer must not Requires= the unit it triggers"
    assert re.search(r"^Unit=.*\.service$", timer, re.MULTILINE), (
        "name the triggered service explicitly rather than relying on the basename default"
    )
    assert re.search(r"^WantedBy=timers\.target$", timer, re.MULTILINE)


def test_only_the_timer_is_enabled():
    """Enabling the oneshot service would push once at boot and never again."""
    assert "[Install]" not in _directives(SERVICE_TEMPLATE), (
        "the sync service must not be enableable; the timer is what gets enabled"
    )
    scheduled = [
        task for task in _tasks()
        if "ansible.builtin.systemd_service" in task
        and task["ansible.builtin.systemd_service"]["name"]
        == "{{ hotkey_sync_service }}.timer"
    ]
    assert len(scheduled) == 1, "exactly one task should enable the hotkey sync timer"
    assert scheduled[0]["ansible.builtin.systemd_service"]["enabled"] is True


# --- 3. the gates, in both directions ------------------------------------


ALIVE = (
    "NextElapseUSecRealtime=\n"
    "NextElapseUSecMonotonic=2w 1d 13h 8min 11.136640s\n"
    "ActiveState=active\n"
    "UnitFileState=enabled\n"
)
DEAD = (
    "NextElapseUSecRealtime=\n"
    "NextElapseUSecMonotonic=infinity\n"
    "ActiveState=inactive\n"
    "UnitFileState=enabled\n"
)
NEVER_FIRES = (
    "NextElapseUSecRealtime=\n"
    "NextElapseUSecMonotonic=infinity\n"
    "ActiveState=active\n"
    "UnitFileState=enabled\n"
)
DISABLED = (
    "NextElapseUSecRealtime=\n"
    "NextElapseUSecMonotonic=2w 1d 13h 8min 11.136640s\n"
    "ActiveState=active\n"
    "UnitFileState=disabled\n"
)


@pytest.mark.parametrize(
    ("task_name", "variable"),
    [
        ("Confirm the schedule is armed and has a next run (Linux)", "slot_manager_schedule"),
        ("Confirm the sync schedule is armed and has a next run (Linux)", "hotkey_sync_schedule"),
    ],
)
def test_the_armed_schedule_gate_can_fail_as_well_as_pass(task_name, variable):
    """These are monotonic timers, so NextElapseUSecRealtime is always EMPTY.

    In production: the purchase timer's gate required the STRING
    ``NextElapseUSecRealtime=`` to be present in ``systemctl show`` output.
    systemd prints that property for every timer, empty for a monotonic one, so
    the check passed for every possible state of the timer including "will never
    fire again" -- a gate that could not fail, reporting that the schedule had a
    next run.
    """
    expression = _task(task_name)["failed_when"]
    assert "NextElapseUSecRealtime" not in expression, (
        "Realtime is empty for OnBootSec/OnUnitActiveSec timers; reading it is "
        "the check that could not fail"
    )

    assert not _evaluate(expression, **{variable: {"stdout": ALIVE}}), (
        "a healthy armed timer must not fail the deploy"
    )
    for label, output in (("stopped", DEAD), ("never fires again", NEVER_FIRES),
                          ("not enabled at boot", DISABLED)):
        assert _evaluate(expression, **{variable: {"stdout": output}}), (
            f"a timer that is {label} must fail this gate"
        )


SCRIPT_OUTCOMES = [
    "no hotkey directory at /home/thomasg/.bittensor/wallets/mining_ops/hotkeys",
    "no hotkeys matching nova68-slot yet",
    "no running instance labelled nova-nanobody-pixi; nothing to sync",
    "synced 6 local file(s); box now holds 6 matching nova68-slot",
]


def test_the_proving_run_accepts_every_healthy_outcome_and_no_others():
    """Three of the script's four outcomes are healthy states of a live fleet.

    Requiring a sync to have HAPPENED would fail every deploy made while no GPU
    is rented, which is most of them: the purchaser exists precisely to keep
    buying when no box is up. What the gate requires is evidence that this unit
    reached one of its own outcomes, which is what separates "ran and had
    nothing to do" from "timer armed in front of something that no longer runs".
    """
    task = _task("Require that run to have reached one of the script's outcomes (Linux)")
    expression = task["ansible.builtin.assert"]["that"][0]

    for outcome in SCRIPT_OUTCOMES:
        journal = f"Starting nova-hotkey-sync.service...\n{outcome}\nFinished.\n"
        assert _evaluate(expression, hotkey_sync_journal={"stdout": journal}), (
            f"a healthy run reporting {outcome!r} must not fail the deploy"
        )

    silent = "Starting nova-hotkey-sync.service...\nFinished.\n"
    assert not _evaluate(expression, hotkey_sync_journal={"stdout": silent}), (
        "a unit that ran and reported none of its outcomes must fail the gate"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Starting nova-hotkey-sync.service...\nFinished.\n",  # no outcome at all
        "",
    ],
)
def test_the_proving_gate_still_prints_its_explanation_when_it_fails(message):
    """Ansible renders success_msg even for a FAILING assert.

    In production: this gate's success_msg was a bare ``regex_search`` that
    returns None when nothing matches, so the one run that should have explained
    itself died with "argument 'success_msg' is of type NoneType" and the
    explanation never reached the operator. Same shape as the failure below it.
    """
    block = _task("Require that run to have reached one of the script's outcomes (Linux)")[
        "ansible.builtin.assert"
    ]
    for key in ("fail_msg", "success_msg"):
        rendered = _render(
            block[key],
            hotkey_sync_journal={"stdout": message},
            hotkey_sync_service="nova-hotkey-sync",
            hotkey_sync_script="/opt/nova-nanobody/deploy/slot_host/sync_hotkeys.sh",
        )
        assert rendered.strip(), f"{key} renders empty when the gate fails"
        assert "None" not in rendered, f"{key} renders None when nothing matched"


def test_the_discovery_gate_still_prints_its_explanation_when_it_fails():
    """A client that answers nothing must produce the explanation, not a TypeError.

    ``/bin/true`` is an executable that exits 0 and says nothing, which is what
    a wrong NOVA_VASTAI looks like. The gate's messages index the probe's output,
    and an unguarded index turned the failure into "object of type 'list' has no
    attribute 1" -- with the deploy still failing, but for the wrong reason and
    without saying what to fix.
    """
    block = _task("Require a vastai client that can answer (Linux)")["ansible.builtin.assert"]
    variables = dict(
        hotkey_sync_instances={"stdout": "", "stderr": "Traceback ..."},
        hotkey_sync_vastai="/bin/true",
        hotkey_sync_vast_label="nova-nanobody-pixi",
        hotkey_sync_api_key_paths=["/home/thomasg/.config/vastai/vast_api_key"],
        slot_manager_user="thomasg",
    )
    for key in ("fail_msg", "success_msg"):
        assert _render(block[key], **variables).strip(), f"{key} renders empty on failure"

    expression = block["that"][0]
    assert not _evaluate(expression, **variables), "an empty answer must fail the gate"
    variables["hotkey_sync_instances"] = {"stdout": "VASTAI_RAW_OK 1 1 0\n", "stderr": ""}
    assert _evaluate(expression, **variables), (
        "a client that answers with no box rented must pass: that is a healthy "
        "purchase host between rentals"
    )


def test_the_deploy_messages_collapse_whitespace_the_ansible_way():
    """A doubled backslash here is a value that is fine until the boundary.

    ``regex_replace('\\s+', ' ')`` renders and silences jinja2's warning about an
    invalid escape sequence, and then matches a literal backslash followed by an
    s. Nothing fails: the filter simply stops collapsing anything, and the next
    failure of a gate puts a 40-line Python traceback into the deploy recap with
    its newlines intact. Verified on the purchase host in both spellings.
    """
    text = SLOTS.read_text()
    assert "regex_replace('\\\\s+'" not in text, (
        "the doubled spelling reaches re.sub as \\\\s+ and matches no whitespace"
    )
    assert "regex_replace('\\s+'" in text, (
        "the deploy's messages no longer collapse multi-line stderr"
    )


def test_a_box_that_comes_back_short_fails_the_deploy():
    """The drift this timer exists to end, stated as an invariant.

    The box may hold MORE than crossed this run -- keys from an earlier wallet
    generation are never deleted -- but it must never hold fewer than what was
    just pushed, and it must never be zero while hotkeys exist locally.
    """
    counts = _task("Count what crossed and what the box holds (Linux)")["ansible.builtin.set_fact"]
    guard = _task("Require the box to hold at least what was pushed (Linux)")[
        "ansible.builtin.assert"
    ]["that"]

    def _verdict(line: str) -> bool:
        journal = {"stdout": f"Starting...\n{line}\n"}
        # The set_fact values are already complete templates; rendering them is
        # what the play does, and it is what turns them into strings that the
        # guard below has to coerce with | int.
        facts = {
            key: _render(value, hotkey_sync_journal=journal)
            for key, value in counts.items()
        }
        return all(_evaluate(item, **facts) for item in guard)

    assert _verdict("synced 6 local file(s); box now holds 6 matching nova68-slot")
    assert _verdict("synced 6 local file(s); box now holds 8 matching nova68-slot"), (
        "a box holding keys from an earlier generation is not a failure"
    )
    assert not _verdict("synced 6 local file(s); box now holds 2 matching nova68-slot"), (
        "a box that came back short is exactly the drift this unit exists to end"
    )
    assert not _verdict("synced 0 local file(s); box now holds 0 matching nova68-slot"), (
        "pushing nothing and finding nothing is not a successful delivery"
    )


# --- 4. the lock must describe the host that runs it ---------------------


def _manifest_dependencies() -> tuple[list[str], list[str]]:
    """(conda, pypi) dependency names from the purchase host's manifest.

    Parsed with a regex rather than a TOML library because this file has to run
    wherever the rest of the suite runs, and the manifest is two flat tables.
    """
    text = (SLOT_HOST / "pixi.toml").read_text()
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections.setdefault(current, [])
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*=", stripped)
        if match and current in {"dependencies", "pypi-dependencies"}:
            sections[current].append(match.group(1))
    return sections.get("dependencies", []), sections.get("pypi-dependencies", [])


def test_the_purchase_host_lock_describes_the_purchase_host():
    """A lock that omits a declared dependency is a lock nobody can install from.

    In production: ``pixi.toml`` declared ``vastai`` -- the client the hotkey
    sync unit shells out to -- and ``pixi.lock`` never mentioned it, because the
    environment had been built on the Pi without ``--locked``. Nothing broke
    while no playbook ran ``pixi install``; the first one that did failed with
    "lock file not up-to-date with the workspace", on the one always-on host
    that spends TAO.
    """
    lock = (SLOT_HOST / "pixi.lock").read_text()
    conda, pypi = _manifest_dependencies()
    assert "vastai" in pypi, "the sync unit's client is no longer declared"

    for name in pypi:
        assert re.search(rf"^  name: {re.escape(name)}$", lock, re.MULTILINE), (
            f"deploy/slot_host/pixi.toml declares the pypi dependency {name!r} but "
            "pixi.lock does not contain it; re-lock ON the purchase host "
            "(`pixi lock` in deploy/slot_host) and commit the result"
        )
    for name in conda:
        assert re.search(rf"/{re.escape(name)}-\d", lock), (
            f"deploy/slot_host/pixi.toml declares the conda dependency {name!r} but "
            "no locked package matches it"
        )


def test_the_lock_is_resolved_for_the_purchase_host_and_nothing_else():
    """arm64 Linux, resolved natively. The macOS controller cannot do it.

    The GPU workspace is linux-64 and this one is linux-aarch64 on purpose: a
    single manifest covering both would force every environment to re-solve,
    including the git-backed builds that cannot be cross-resolved at all.
    """
    lock = yaml.safe_load((SLOT_HOST / "pixi.lock").read_text())
    assert [platform["name"] for platform in lock["platforms"]] == ["linux-aarch64"]
    assert re.search(
        r'^platforms = \["linux-aarch64"\]$',
        (SLOT_HOST / "pixi.toml").read_text(),
        re.MULTILINE,
    )


def test_the_purchase_host_environment_is_built_from_the_lock():
    """The interpreter the purchaser runs must not be a hand-built artefact.

    It was one: the gate below it proved that ``bittensor`` imports, but nothing
    could recreate the environment it imported from, so a reprovisioned Pi armed
    a timer in front of an interpreter that did not exist.
    """
    build = _task("Build the purchase host's environment from the committed lock (Linux)")
    argv = [str(item) for item in build["ansible.builtin.command"]["argv"]]
    assert argv[1:] == ["install", "--locked"], (
        "a bare `pixi install` resolves whatever is newest and rewrites the lock "
        "on a live spending host; --locked makes a drifted lock fail loudly"
    )
    assert build["ansible.builtin.command"]["chdir"] == "{{ slot_host_workspace }}"
    assert build.get("become") is not True, (
        "pixi must run as the service user, or .pixi becomes root-owned and the "
        "unit cannot rebuild it"
    )

    text = SLOTS.read_text()
    for forbidden in ("--frozen", "pixi, add", "pixi add"):
        assert forbidden not in text, f"{forbidden} defeats the point of the lock"

    # The build has to precede both checks that use the interpreter it creates.
    names = [task.get("name") for task in _tasks()]
    assert names.index(build["name"]) < names.index(
        "Require the chain stack the agent needs for its reads"
    )


# --- 5. credentials stay locators ---------------------------------------


def test_the_api_key_is_located_and_never_read():
    """Only the PATH of the vast.ai key may appear, and only under stat.

    The coldkey password reaches the purchaser through LoadCredentialEncrypted so
    that it appears in no unit, no `systemctl show` and no /proc/<pid>/environ.
    The same rule applies to the key the sync unit's client needs: the client
    resolves it itself, and the deployment only ever asks whether the file is
    there.
    """
    stat = _task("Locate the vast.ai API key the sync unit needs (Linux)")["ansible.builtin.stat"]
    assert stat["get_checksum"] is False, (
        "stat hashes file contents by default; the key's bytes must not be read"
    )

    text = SLOTS.read_text()
    for reader in ("ansible.builtin.slurp", "lookup('file'", 'lookup("file"',
                   "ansible.builtin.fetch"):
        assert reader not in text, f"{reader} could read a credential into the controller"

    for template in (SERVICE_TEMPLATE, TIMER_TEMPLATE):
        body = _directives(template)
        assert "vast_api_key" not in body, f"{template.name} names the key file"
        assert "LoadCredential" not in body, (
            f"{template.name} has no business with the coldkey credential"
        )

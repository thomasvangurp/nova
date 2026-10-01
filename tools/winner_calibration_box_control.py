#!/usr/bin/env python3
"""Guard, retrieve, analyze, report, and stop the approved offline replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "analysis/nanobody-gap/winner-calibration-ten-ticket-20260922"
STATE = DEST / "supervisor_state.json"
INSTANCE_ID = 52073739
LABEL = "nova-nanobody-h2-fast-20260922"
GPU_COUNT = 8
EXPECTED_COUNT = 519
PREREG_SHA = "832882b5be45d7ae0823be1aa278ef9baeb5c2ec03ddf61bfc745d9f8c8bc0b8"
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
REMOTE = "/root/winner-calibration-ten-ticket-20260922"
VAST_API_SERVICE = "vast.ai"
_SSH_ENDPOINT: tuple[str, int] | None = None
REPLACEMENT_MANIFEST = DEST / "replacement_instance.json"


def expected_instance() -> tuple[int, str]:
    if REPLACEMENT_MANIFEST.is_file():
        payload = json.loads(REPLACEMENT_MANIFEST.read_text())
        return int(payload["id"]), str(payload["label"])
    return INSTANCE_ID, LABEL


def write_state(**values: object) -> None:
    prior = json.loads(STATE.read_text()) if STATE.is_file() else {}
    STATE.write_text(json.dumps({
        **prior, **values, "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n")


def keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("credential_unavailable")
    return result.stdout.decode().strip()


def api(path: str, body: object | None = None, method: str = "GET") -> dict:
    credential = keychain(VAST_API_SERVICE)
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/" + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"},
    )
    credential = ""
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def guarded_instance() -> dict:
    expected_id, expected_label = expected_instance()
    matches = [
        item for item in api("instances/?owner=me").get("instances", [])
        if item.get("id") == expected_id
    ]
    if len(matches) != 1:
        raise RuntimeError("instance_not_found")
    instance = matches[0]
    if instance.get("label") != expected_label or instance.get("num_gpus") != GPU_COUNT or instance.get("gpu_name") != "RTX 4090":
        raise RuntimeError("instance_guard_failed")
    return instance


def resume() -> None:
    instance = guarded_instance()
    if instance.get("actual_status") == "running":
        write_state(stage="running")
        return
    expected_id, _ = expected_instance()
    result = api("instances/" + str(expected_id) + "/", {"state": "running"}, "PUT")
    if result.get("success") is not True:
        raise RuntimeError("resume_failed")
    write_state(stage="resume_requested")


def ssh_endpoint() -> tuple[str, int]:
    global _SSH_ENDPOINT
    if _SSH_ENDPOINT is not None:
        return _SSH_ENDPOINT
    instance = guarded_instance()
    if instance.get("actual_status") != "running":
        raise RuntimeError("instance_not_running")
    host = instance.get("ssh_host") or instance.get("public_ipaddr")
    port = instance.get("ssh_port")
    if not port:
        entries = (instance.get("ports") or {}).get("22/tcp") or []
        port = next((row.get("HostPort") for row in entries if row.get("HostPort")), None)
    if not host or not port:
        raise RuntimeError("ssh_endpoint_unavailable")
    _SSH_ENDPOINT = (str(host), int(port))
    return _SSH_ENDPOINT


def ssh() -> list[str]:
    _, port = ssh_endpoint()
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(port),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
    ]


def remote_json(code: str, timeout: int = 60) -> dict | list:
    host, _ = ssh_endpoint()
    result = subprocess.run(
        [*ssh(), "root@" + host, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("remote_failed")
    return json.loads(result.stdout)


def remote_status() -> dict:
    code = f"""
import json,os
from pathlib import Path
base=Path({REMOTE!r});rows=[]
for i in range(17,33):
 d=base/f'fresh_{{i:02d}}';wave=d/'wave_000_input.json';exact=d/'exact_all.json';row={{'draw':i,'started':wave.is_file(),'complete':exact.is_file()}}
 if wave.is_file():row['started_unix']=wave.stat().st_mtime
 if exact.is_file():
  payload=json.loads(exact.read_text());row.update(n=payload.get('n'),protocol=payload.get('scorer_protocol'),completed_unix=exact.stat().st_mtime)
  if wave.is_file():row['duration_seconds']=exact.stat().st_mtime-wave.stat().st_mtime
 rows.append(row)
runner=False
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:args=open(f'/proc/{{pid}}/cmdline','rb').read().split(b'\\0')
 except OSError:continue
 if any(x.endswith(b'/winner-calibration-ten-ticket-20260922/run_remote.sh') for x in args):runner=True
print(json.dumps({{'draws':rows,'complete':(base/'COMPLETE').is_file(),'runner_alive':runner}}))
"""
    return dict(remote_json(code))


def remote_manifest() -> list[dict]:
    code = f"""
import hashlib,json
from pathlib import Path
base=Path({REMOTE!r});paths=sorted(base.glob('fresh_*/exact_all.json'))
paths += [p for p in (base/'run_metadata.json',base/'preregistration.json',base/'sequences.json') if p.is_file()]
rows=[]
for path in paths:
 data=path.read_bytes();rows.append({{'file':str(path.relative_to(base)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime':path.stat().st_mtime}})
print(json.dumps(rows))
"""
    result = remote_json(code)
    if not isinstance(result, list):
        raise RuntimeError("invalid_remote_manifest")
    return result


def fetch() -> dict:
    host, _ = ssh_endpoint()
    status = remote_status()
    rows = remote_manifest()
    allowed = re.compile(r"(?:fresh_(?:1[7-9]|2[0-9]|3[0-2])/exact_all\.json|run_metadata\.json|preregistration\.json|sequences\.json)")
    if hashlib.sha256((DEST / "preregistration.json").read_bytes()).hexdigest() != PREREG_SHA:
        raise RuntimeError("local_preregistration_changed")
    for item in rows:
        relative = str(item.get("file", ""))
        if not allowed.fullmatch(relative):
            raise RuntimeError("unsafe_artifact_name")
        if relative in {"preregistration.json", "sequences.json"}:
            if hashlib.sha256((DEST / relative).read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("remote_frozen_input_mismatch")
            continue
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "rsync", "-rlt", "--checksum", "--no-owner", "--no-group", "--no-perms",
            "-e", shlex.join(ssh()), f"root@{host}:{REMOTE}/{relative}", str(target),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if not (
            result.returncode == 0 and target.is_file()
            and target.stat().st_size == int(item["bytes"])
            and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        ):
            raise RuntimeError("artifact_verification_failed")
    exact = [row for row in rows if row["file"].startswith("fresh_")]
    if exact:
        timings = status["draws"]
        complete = [row for row in timings if row.get("complete")]
        metadata = json.loads((DEST / "run_metadata.json").read_text())
        metadata["draw_timings"] = timings
        if complete:
            metadata["first_wave_started_unix"] = timings[0].get("started_unix")
            metadata["last_draw_completed_unix"] = complete[-1]["completed_unix"]
            metadata["scoring_wall_seconds"] = complete[-1]["completed_unix"] - timings[0]["started_unix"]
            metadata["estimated_scoring_cost"] = metadata["scoring_wall_seconds"] / 3600 * metadata["hourly_cost"]
        (DEST / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"verified_exact": len(exact), "verified_total": len(rows)}


def render_report() -> None:
    payload = json.loads((DEST / "analysis.json").read_text())
    calibration = payload["test1_historical_calibration"]
    test2 = payload["test2_literal_ten_ticket"]
    comparison = test2["comparison"]
    a = test2["A"]["totals"]
    b = test2["B"]["totals"]
    runtime = payload["runtime"]
    lines = [
        "# Historical calibration and literal ten-ticket A/B",
        "",
        f"Overall fail-closed decision: **{payload['overall_decision']}**.",
        "",
        "## Test 1 — historical winner recovery and calibration",
        "",
        f"- Decision: **{'PASS' if calibration['pass'] else 'FAIL'}**",
        f"- Actual winner recovered first: {calibration['winner_recovered_groups']}/{calibration['winning_epoch_groups']} groups",
        f"- IPW predicted win rate: {calibration['weighted_predicted_win_rate']:.4%}; full-history target: {calibration['target_full_history_rate']:.4%}",
        f"- Predicted-rate 95% interval: {calibration['bootstrap']['predicted_rate_95_interval'][0]:.4%} to {calibration['bootstrap']['predicted_rate_95_interval'][1]:.4%}",
        f"- Winner-minus-control prediction: {calibration['winner_minus_control']:.4%}; 95% interval {calibration['bootstrap']['winner_minus_control_95_interval'][0]:.4%} to {calibration['bootstrap']['winner_minus_control_95_interval'][1]:.4%}",
        f"- Weighted Brier: {calibration['weighted_brier']:.6f}; weighted AUC: {calibration['weighted_auc']:.4f}",
        f"- Gates: `{json.dumps(calibration['gates'], sort_keys=True)}`",
        "",
        "## Test 2 — literal ten-ticket A/B",
        "",
        f"- Decision: **{comparison['decision']}**",
        f"- A wins/entry: {a['wins']}/{a['opportunities']} = {a['individual_win_per_entry']:.4%}; B: {b['wins']}/{b['opportunities']} = {b['individual_win_per_entry']:.4%}",
        f"- A top10/entry: {a['top10']}/{a['opportunities']} = {a['top10_per_entry']:.4%}; B: {b['top10']}/{b['opportunities']} = {b['top10_per_entry']:.4%}",
        f"- Favorable draw blocks: wins {comparison['win_favorable_draws']}/16; top10 {comparison['top10_favorable_draws']}/16",
        f"- Panel deltas: `{json.dumps(comparison['panel_deltas'], sort_keys=True)}`",
        f"- Gates: `{json.dumps(comparison['gates'], sort_keys=True)}`",
        "",
        "## Runtime and provenance",
        "",
        f"- Slowest 519-sequence draw: {runtime['slowest_draw_seconds']:.3f}s",
        f"- Scoring sum: {runtime['research_scoring_seconds_sum']:.3f}s; estimated scoring cost: ${runtime['research_scoring_cost']:.3f}",
        f"- Preregistration SHA-256: `{payload['preregistration_sha256']}`",
        "- All field comparisons use intact ten-metric vectors and whole-fleet joint dense ranks.",
        "",
    ]
    (DEST / "REPORT.md").write_text("\n".join(lines))


def analyze() -> None:
    result = subprocess.run(
        [sys.executable, str(DEST / "analyze_fresh.py")], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600,
    )
    if result.returncode:
        raise RuntimeError("analysis_failed")
    render_report()
    paths = sorted(DEST.glob("fresh_*/exact_all.json")) + [
        DEST / "run_metadata.json", DEST / "analysis.json", DEST / "REPORT.md",
    ]
    (DEST / "RESULTS.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(DEST)}\n"
        for path in paths
    ))


def stop() -> None:
    instance = guarded_instance()
    if instance.get("actual_status") not in {"stopped", "exited"}:
        expected_id, _ = expected_instance()
        result = api("instances/" + str(expected_id) + "/", {"state": "stopped"}, "PUT")
        if result.get("success") is not True:
            raise RuntimeError("stop_failed")
    deadline = time.time() + 180
    while time.time() < deadline:
        status = guarded_instance().get("actual_status")
        if status in {"stopped", "exited"}:
            write_state(stage="stopped", instance_status=status)
            return
        time.sleep(5)
    raise RuntimeError("stop_not_confirmed")


def monitor() -> int:
    deadline = time.time() + 4 * 60 * 60
    previous_complete = -1
    write_state(stage="monitoring", deadline_unix=deadline)
    while time.time() < deadline:
        remote = remote_status()
        complete_count = sum(row.get("complete", False) for row in remote["draws"])
        write_state(stage="monitoring", **remote)
        if complete_count != previous_complete:
            completed = [row for row in remote["draws"] if row.get("complete")]
            print(json.dumps({"completed": complete_count, "latest": completed[-1] if completed else None}), flush=True)
            previous_complete = complete_count
        if remote["complete"]:
            summary = fetch()
            if summary["verified_exact"] != 16:
                raise RuntimeError("incomplete_artifact_set")
            for draw in range(17, 33):
                payload = json.loads((DEST / f"fresh_{draw:02d}" / "exact_all.json").read_text())
                if payload.get("scorer_protocol") != "validator-dual-sample-v1" or int(payload.get("n", -1)) != EXPECTED_COUNT:
                    raise RuntimeError("invalid_exact_artifact")
            analyze()
            write_state(stage="analyzed", **summary)
            stop()
            return 0
        if not remote["runner_alive"]:
            summary = fetch()
            write_state(stage="runner_failed_box_held", **summary)
            return 1
        time.sleep(20)
    summary = fetch()
    write_state(stage="deadline_box_held", **summary)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("resume", "status", "fetch", "monitor", "stop"))
    operation = parser.parse_args().operation
    if operation == "resume":
        resume()
    elif operation == "status":
        instance = guarded_instance()
        safe = {key: instance.get(key) for key in ("id", "label", "actual_status", "num_gpus", "gpu_name")}
        try:
            remote = remote_status()
        except Exception as error:
            remote = {"error_class": type(error).__name__}
        print(json.dumps({"instance": safe, "remote": remote}, sort_keys=True))
    elif operation == "fetch":
        print(json.dumps(fetch(), sort_keys=True))
    elif operation == "stop":
        stop()
    else:
        return monitor()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_state(stage="failed_box_held", error_class=type(error).__name__)
        print(json.dumps({"event": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1)

"""Fund infrastructure from mined TAO with bounded, idempotent chain writes.

The destination is local configuration on the Pi, never supplied by the daily
instruction.  The instruction can vary the amount only inside three guards:
current operating cost, measured mining earnings, and a hard TAO/day ceiling.
The coldkey retains a separate free-TAO registration reserve.  If necessary,
only enough SN68 alpha to restore that reserve plus the transfer is converted,
using the slot manager's safe unstaking path and slippage limit.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from .slot_manager import (
    _alpha_price,
    _import_bittensor,
    _unlock_coldkey,
    top_up_tao,
)


MINIMUM_DEPOSIT_TAO = 0.01


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _records(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def decide(target: dict, records: list[dict], *, now: float,
           maximum_tao: float) -> dict:
    """Validate a daily instruction and refuse duplicates or uncertainty."""
    today = dt.datetime.fromtimestamp(now, dt.timezone.utc).date().isoformat()
    if int(target.get("schema_version") or 0) != 1:
        return {"action": "wait", "reason": "unsupported_target_schema"}
    if str(target.get("target_day") or "") != today:
        return {"action": "wait", "reason": "target_not_for_today"}
    if int(target.get("generated_at") or 0) > now + 300:
        return {"action": "wait", "reason": "target_from_future"}
    if int(target.get("valid_until") or 0) < now:
        return {"action": "wait", "reason": "target_expired"}
    amount = float(target.get("authorized_tao") or 0)
    if amount < MINIMUM_DEPOSIT_TAO:
        return {"action": "wait", "reason": "below_deposit_minimum"}
    if amount > float(maximum_tao) + 1e-12:
        return {"action": "wait", "reason": "hard_cap_exceeded"}
    if amount > float(target.get("cost_tao") or 0) + 1e-9:
        return {"action": "wait", "reason": "cost_cap_exceeded"}
    if amount > float(target.get("earnings_tao_per_day") or 0) + 1e-9:
        return {"action": "wait", "reason": "earnings_cap_exceeded"}
    same_day = [r for r in records if r.get("target_day") == today]
    if any(r.get("state") in {"transfer_started", "complete", "uncertain"}
           for r in same_day):
        return {"action": "wait", "reason": "already_attempted_today"}
    return {"action": "transfer", "reason": "fresh_bounded_target",
            "amount_tao": amount, "target_day": today}


def _safe_result(**values: Any) -> dict:
    allowed = {
        "ok", "action", "reason", "state", "target_day", "amount_tao",
        "fee_tao", "free_before_tao", "free_after_tao", "alpha_converted",
        "destination_suffix", "block_hash", "extrinsic_hash", "error_type",
    }
    return {key: value for key, value in values.items() if key in allowed}


def _tao(value: Any) -> float:
    """Normalize a bittensor Balance without confusing rao with TAO."""
    return float(getattr(value, "tao", value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("/etc/nova/infra-sweep.json"))
    parser.add_argument("--target", type=Path,
                        default=Path.home() / ".nova/infra_funding_target.json")
    parser.add_argument("--ledger", type=Path,
                        default=Path.home() / ".nova/infra_sweeps.jsonl")
    parser.add_argument("--wallet", default="mining_ops")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--keychain-service-env",
                        default="NOVA_COLDKEY_KEYCHAIN_SERVICE")
    parser.add_argument("--credential-source", choices=("auto", "keychain", "systemd"),
                        default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    target = _read_json(args.target)
    maximum_tao = float(config.get("maximum_tao_per_day") or 0)
    decision = decide(target, _records(args.ledger), now=time.time(),
                      maximum_tao=maximum_tao)
    if decision["action"] != "transfer":
        print(json.dumps(_safe_result(ok=True, **decision), sort_keys=True))
        return 0

    destination = str(config.get("destination_ss58") or "").strip()
    reserve_tao = float(config.get("registration_reserve_tao") or 0)
    slippage = float(config.get("convert_slippage") or 0.005)
    max_alpha = float(config.get("maximum_convert_alpha_per_day") or 100)
    amount_tao = float(decision["amount_tao"])
    suffix = destination[-8:] if destination else "missing"
    bittensor = _import_bittensor()
    if not bittensor.utils.is_valid_ss58_address(destination):
        print(json.dumps(_safe_result(
            ok=False, action="wait", reason="invalid_configured_destination",
            destination_suffix=suffix,
        ), sort_keys=True))
        return 1

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            wallet_factory = getattr(bittensor, "Wallet", None) or bittensor.wallet
            wallet = wallet_factory(name=args.wallet)
            subtensor_factory = getattr(bittensor, "Subtensor", None) or bittensor.subtensor
            subtensor = subtensor_factory(network=args.network)
            free_before = _tao(
                subtensor.get_balance(wallet.coldkeypub.ss58_address)
            )
            balance = bittensor.Balance.from_tao(amount_tao)
            fee = _tao(subtensor.get_transfer_fee(
                wallet, destination, balance, keep_alive=True,
            ))
        needed = reserve_tao + amount_tao + fee
        # A shortfall below the subnet minimum unstake amount cannot be
        # converted back from alpha, and the reserve is a safety margin
        # rather than a hard floor: tolerate a small dip instead of
        # failing the whole sweep over a sub-cent residual.
        threshold = needed - 0.02
        converted = 0.0
        if free_before < threshold:
            if not args.execute:
                price = _alpha_price(subtensor, args.netuid)
                converted = max(0.0, needed - free_before) / price if price > 0 else 0.0
            else:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    _unlock_coldkey(
                        wallet, args.keychain_service_env, args.credential_source,
                    )
                    outcome = top_up_tao(
                        subtensor, wallet, netuid=args.netuid,
                        coldkey=wallet.coldkeypub.ss58_address,
                        need_tao=free_before, target_tao=needed,
                        price_tao=_alpha_price(subtensor, args.netuid),
                        slippage=slippage, daily_cap_alpha=max_alpha,
                        spent_alpha_today=0.0, execute=True,
                    )
                    converted = float(outcome.get("alpha_converted") or 0)
                    free_before = _tao(
                        subtensor.get_balance(wallet.coldkeypub.ss58_address)
                    )
        if not args.execute:
            print(json.dumps(_safe_result(
                ok=True, action="transfer", reason="dry_run",
                target_day=decision["target_day"], amount_tao=amount_tao,
                fee_tao=round(fee, 9), free_before_tao=round(free_before, 9),
                alpha_converted=round(converted, 6),
                destination_suffix=suffix,
            ), sort_keys=True))
            return 0
        if free_before < threshold:
            print(json.dumps(_safe_result(
                ok=False, action="wait", reason="reserve_not_funded",
                target_day=decision["target_day"], amount_tao=amount_tao,
                fee_tao=round(fee, 9), free_before_tao=round(free_before, 9),
                alpha_converted=round(converted, 6),
                destination_suffix=suffix,
            ), sort_keys=True))
            return 1

        # A transfer needs the coldkey even when no alpha conversion was
        # necessary.  Calling this twice is harmless: an already decrypted
        # keyfile returns immediately.
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            _unlock_coldkey(
                wallet, args.keychain_service_env, args.credential_source,
            )

        _append(args.ledger, {
            "state": "transfer_started", "target_day": decision["target_day"],
            "at": int(time.time()), "amount_tao": amount_tao,
            "destination_suffix": suffix,
        })
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            response = subtensor.transfer(
                wallet=wallet, destination_ss58=destination, amount=balance,
                transfer_all=False, keep_alive=True, raise_error=True,
                wait_for_inclusion=True, wait_for_finalization=False,
            )
            success = bool(getattr(response, "success", response))
            free_after = _tao(
                subtensor.get_balance(wallet.coldkeypub.ss58_address)
            )
        state = "complete" if success else "uncertain"
        record = {
            "state": state, "target_day": decision["target_day"],
            "at": int(time.time()), "amount_tao": amount_tao,
            "fee_tao": round(fee, 9), "free_before_tao": round(free_before, 9),
            "free_after_tao": round(free_after, 9),
            "alpha_converted": round(converted, 6),
            "destination_suffix": suffix,
            "block_hash": str(getattr(response, "block_hash", "") or ""),
            "extrinsic_hash": str(getattr(response, "extrinsic_hash", "") or ""),
        }
        _append(args.ledger, record)
        print(json.dumps(_safe_result(ok=success, action="transfer", **record),
                         sort_keys=True))
        return 0 if success else 1
    except BaseException as error:
        result = _safe_result(
            ok=False, action="wait", reason="execution_error",
            target_day=decision.get("target_day"), amount_tao=amount_tao,
            destination_suffix=suffix, error_type=type(error).__name__,
        )
        print(json.dumps(result, sort_keys=True))
        return 1
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())

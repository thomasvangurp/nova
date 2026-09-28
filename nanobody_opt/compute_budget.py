"""Wall-time plans for screening and independent confirmation on the epoch clock."""
from __future__ import annotations

from dataclasses import dataclass
import math


def _ran_on(row: dict, default: int) -> int:
    """GPU count a timing record was measured on, else the current count."""
    try:
        count = int(row.get("gpu_count") or 0)
    except (TypeError, ValueError):
        count = 0
    return count if count > 0 else default


@dataclass(frozen=True)
class TimingModel:
    load_seconds: float = 45.0
    gpu_seconds_per_candidate: float = 7.5
    cpu_guard_seconds: float = 120.0

    #: A fitted model may narrow, but never past physical sense.
    HARD_FLOOR_LOAD = 1.0
    HARD_FLOOR_GPU = 0.25

    #: How far measurement may inflate the configured cost. A TRANSIENT
    #: slowdown must not be able to collapse the shortlist for a whole
    #: epoch. Epochs 25239/25240: a ~40% screen slowdown lasting minutes
    #: raised gpu_seconds_per_candidate 5.5 -> 8.08 and cut confirmation
    #: width 60 -> 40, and because the estimator only ever rose it never
    #: recovered inside the epoch. Beyond this multiple the right response
    #: is to keep the configured plan, not to shrink the tournament.
    MAX_INFLATION = 2.0

    def round_seconds(self, candidates: int, gpus: int) -> float:
        if candidates < 1 or gpus < 1:
            raise ValueError("positive candidate and GPU counts are required")
        return self.load_seconds + self.gpu_seconds_per_candidate * math.ceil(candidates / gpus)

    def staged_seconds(self, *, rounds: int, candidates: int, gpus: int,
                       rounds_per_stage: int = 0,
                       keep_fraction: float = 1.0) -> float:
        """Cost of a replication run that PRUNES between stages.

        `rounds * round_seconds(width)` charges every round at the full
        shortlist, but `_run_replication_rounds` runs successive elimination:
        after each stage it keeps `max(int(alive * keep_fraction), 1)`. At
        width 80, keep 0.6, 4 rounds per stage, the real sequence is
        80 -> 48 -> 28, costing ~1038 s where the flat form charges ~1500 s.
        The planner reserved that difference every epoch and never spent it.

        This mirrors the loop's arithmetic exactly, including the truncating
        int() and the floor of one survivor.
        """
        if rounds < 1 or candidates < 1 or gpus < 1:
            raise ValueError("positive rounds, candidates and GPUs are required")
        total = 0.0
        alive = int(candidates)
        remaining = int(rounds)
        while remaining > 0:
            this_stage = min(rounds_per_stage or remaining, remaining)
            total += this_stage * self.round_seconds(alive, gpus)
            remaining -= this_stage
            if keep_fraction < 1.0 and alive > 1:
                alive = max(int(alive * keep_fraction), 1)
        return total

    #: Only these rounds describe the cost being planned. Screen tranches
    #: fold hundreds of candidates in one call and replication rounds fold a
    #: handful; pooling them fits a line through two unrelated regimes. On
    #: live records that pooling pushed gpu_seconds_per_candidate from 5.5 to
    #: 16.0 and throttled the shortlist to 20.
    CONFIRMATION_PHASES = ("replication_round",)

    def fit_from_measurements(self, records: list[dict], gpus: int, *,
                              window: int = 80, quantile: float = 0.90,
                              safety: float = 1.15,
                              minimum_samples: int = 20,
                              phases: tuple = CONFIRMATION_PHASES) -> TimingModel:
        """Fit the model to recent rounds, in BOTH directions.

        `widen_for_measurements` can only raise, and starts from the CLI
        default every epoch, so the default is a permanent floor: a 45 s
        constant survived two days of evidence that the real figure was
        ~1.2 s, and no amount of measurement could correct it. It also uses
        `max` over every record, so one contended round pins the estimate
        for the whole epoch.

        Instead: least squares over the last `window` rounds, then pad the
        constant by the p90 residual so ~90% of rounds fit the plan rather
        than 100% including the pathological one. Planning for the very
        worst round is paying twice, because the run loop ALREADY checks the
        clock before each round and stops with EXACT_REPLICATION_FINISH_GUARD.
        An optimistic plan therefore costs completed rounds, never the epoch.

        Narrowing needs `minimum_samples`; below that this defers to the
        widen-only path, so a thin or unusual epoch stays conservative.
        """
        rows = []
        for row in records:
            n = row.get("candidates", row.get("shortlist", 0))
            seconds = row.get("scoring_seconds", row.get("seconds", 0))
            if row.get("phase") in phases and n and seconds:
                # A round's cost follows the GPU count it RAN on. Seed epochs
                # from a 4-GPU box divided by 8 doubled the per-candidate
                # estimate after the 8x migration (9.37 vs 5.5 measured).
                rows.append((math.ceil(n / _ran_on(row, gpus)), float(seconds)))
        rows = rows[-window:]
        if len(rows) < minimum_samples:
            # NOT a max over whatever happens to exist. Early in an epoch --
            # which is when the shortlist is decided -- there are only a
            # handful of rows, and taking their max let one unlucky round
            # set the plan. Widen from a high quantile of them instead, and
            # never past MAX_INFLATION of the configured cost.
            return self._bounded_widen(rows, records, gpus, quantile, safety)
        n = len(rows)
        # Theil-Sen, not least squares. A single 600 s round at the narrowest
        # width drags an OLS slope negative, which tripped the "nonsense fit"
        # guard and fell back to the widen-only model -- so one outlier
        # produced the very worst-case estimate, the opposite of the intent.
        # The median of pairwise slopes ignores it.
        widened = self._bounded_widen(rows, records, gpus, quantile, safety)
        slopes = sorted((yj - yi) / (xj - xi)
                        for i, (xi, yi) in enumerate(rows)
                        for xj, yj in rows[i + 1:]
                        if xj != xi)
        if not slopes:                  # every round the same width: no slope
            return widened
        slope = slopes[len(slopes) // 2]
        if slope <= 0:                  # genuinely no relationship; stay safe
            return widened
        intercepts = sorted(y - slope * x for x, y in rows)
        intercept = intercepts[len(intercepts) // 2]
        residuals = sorted(y - (intercept + slope * x) for x, y in rows)
        pad = residuals[min(int(quantile * (n - 1)), n - 1)]
        return TimingModel(
            self._clamp(safety * (intercept + max(pad, 0.0)),
                        self.HARD_FLOOR_LOAD,
                        self.MAX_INFLATION * self.load_seconds),
            self._clamp(safety * slope, self.HARD_FLOOR_GPU,
                        self.MAX_INFLATION * self.gpu_seconds_per_candidate),
            widened.cpu_guard_seconds,  # the CPU guard stays widen-only
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(max(value, low), max(low, high))

    def _bounded_widen(self, rows, records, gpus, quantile, safety) -> TimingModel:
        """Widen from a QUANTILE of the planned phase, within MAX_INFLATION.

        The historical estimator took a max over every record and could rise
        without limit, so a transient contention spike throttled the
        shortlist for the rest of the epoch and could not recover.
        """
        cpu = self.cpu_guard_seconds
        for row in records:
            seconds = row.get("scoring_seconds", row.get("seconds", 0))
            if row.get("phase") in {"layer2_gate", "producer_preparation"} and seconds:
                cpu = max(cpu, 1.15 * seconds)
        if not rows:
            return TimingModel(self.load_seconds,
                               self.gpu_seconds_per_candidate, cpu)
        singles = sorted(y for x, y in rows if x <= 1)
        load = self.load_seconds
        if singles:
            picked = singles[min(int(quantile * (len(singles) - 1)),
                                 len(singles) - 1)]
            load = self._clamp(max(load, safety * picked), self.HARD_FLOOR_LOAD,
                               self.MAX_INFLATION * self.load_seconds)
        rates = sorted(max(y - load, 0.0) / x for x, y in rows if x)
        gpu = self.gpu_seconds_per_candidate
        if rates:
            picked = rates[min(int(quantile * (len(rates) - 1)), len(rates) - 1)]
            gpu = self._clamp(max(gpu, safety * picked), self.HARD_FLOOR_GPU,
                              self.MAX_INFLATION * self.gpu_seconds_per_candidate)
        return TimingModel(load, gpu, cpu)

    def widen_for_measurements(self, records: list[dict], gpus: int, *,
                               phases: tuple = CONFIRMATION_PHASES) -> TimingModel:
        """Raise validated bounds when actual work is slower; never learn optimism.

        SCREEN TRANCHES MUST NOT FEED THIS. Pooling them cost epochs 25239
        and 25240. A screen tranche folds ~700 candidates in one call; a
        replication round folds tens. Charging the tranche's seconds against
        `ceil(n/gpus)` produced a per-candidate figure for a different
        regime entirely, and because this only ever raises, one slow tranche
        throttled the confirmation shortlist for the rest of the epoch:

            25238  1.15*(796.5-45)/175 = 4.939  < 5.5 default -> width 60
            25239  1.15*(1105.5-45)/175 = 6.969            -> width 48
            25240  1.15*(438.3-45)/56   = 8.077            -> width 40

        The 5.5 default masked it for thirteen epochs; it bit the moment
        screen cost crossed the default. Width is decided early in an epoch,
        when replication rows are too few to fit, so THIS path is the one
        that usually decides it -- restricting only the fitted path left the
        bug live where it mattered.
        """
        load, gpu, cpu = self.load_seconds, self.gpu_seconds_per_candidate, self.cpu_guard_seconds
        for row in records:
            n = row.get("candidates", row.get("shortlist", 0))
            seconds = row.get("scoring_seconds", row.get("seconds", 0))
            if row.get("phase") in phases and n and seconds:
                ran_on = _ran_on(row, gpus)
                if n <= ran_on:
                    load = max(load, 1.15 * seconds)
                gpu = max(gpu, 1.15 * max(seconds - load, 0) / math.ceil(n / ran_on))
            if row.get("phase") in {"layer2_gate", "producer_preparation"} and seconds:
                cpu = max(cpu, 1.15 * seconds)
        return TimingModel(load, gpu, cpu)


def confirmation_plan(*, epoch_length: int, commitment_blocks: int, rounds: int,
                      maximum_candidates: int, gpus: int, model: TimingModel,
                      time_share: float = .25, block_seconds: float = 12.0,
                      rounds_per_stage: int = 0, keep_fraction: float = 1.0) -> dict:
    if not 0 < time_share < 1 or rounds < 1 or maximum_candidates < 1 or gpus < 1:
        raise ValueError("invalid confirmation plan")
    usable = max(epoch_length - commitment_blocks, 0) * block_seconds
    available = max(usable - model.cpu_guard_seconds, 0)
    budget = min(available, max(usable * time_share, rounds * model.round_seconds(1, gpus)))

    def cost(candidates: int) -> float:
        return model.staged_seconds(
            rounds=rounds, candidates=candidates, gpus=gpus,
            rounds_per_stage=rounds_per_stage, keep_fraction=keep_fraction)

    width = maximum_candidates
    while width > 1 and cost(width) > budget:
        width -= 1
    # Reserve what the plan COSTS, not the whole time share. Charging the
    # full allotment while the staged run spends about two thirds of it left
    # ~35% of the reserve unused every epoch, and those blocks come straight
    # out of screening.
    # The reserve stays the full allotment ON PURPOSE. The ~35% that used to
    # go unspent is now recovered by WIDTH: pricing the staged run honestly
    # lets the same blocks buy a wider shortlist, instead of handing blocks
    # back to screening and making confirmation start later. Starting later
    # is the worse failure: an under-estimate then truncates the LAST stage,
    # which is the narrow, most informative one.
    planned = cost(width)
    reserve = math.ceil((budget + model.cpu_guard_seconds) / block_seconds)
    return {"confirmation_candidates": width, "confirmation_rounds": rounds,
            "replication_reserve_blocks": min(reserve, max(epoch_length - commitment_blocks, 0)),
            "screen_cutoff_block": max(epoch_length - commitment_blocks - reserve, 0),
            "confirmation_budget_seconds": budget,
            "confirmation_planned_seconds": round(planned, 1),
            "time_share": time_share,
            "load_seconds": model.load_seconds,
            "gpu_seconds_per_candidate": model.gpu_seconds_per_candidate,
            "cpu_guard_seconds": model.cpu_guard_seconds}

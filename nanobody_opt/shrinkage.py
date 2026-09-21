"""Empirical-Bayes shrinkage for unevenly-measured candidates.

WHY THIS EXISTS. The screen ranks every candidate on whatever observations
that epoch happened to produce, which is a single fold for most of them. But
the challenge never changes -- one config md5 across every epoch, one target
(P20809) across all 54,719 nodes in the mutation database -- so a fold made
in an earlier epoch is still valid evidence today. Measured on the epoch
25250 pool: 1,316 sequences, mean 1.03 observations *in that epoch* and mean
39.8 across all epochs. We were ranking on 2.6% of the evidence we held.

Using it naively would be wrong in the other direction. Depth is not uniform:
a sequence has forty folds because we shortlisted it before, and a brand new
proposal has one. Comparing a 1-fold mean against a 40-fold mean as if they
were equally trustworthy is what produces the winner's curse already visible
in our submissions -- our selected entries rank *worse* than the field median
(133 against 116), which is what selecting on the largest measurement error
looks like.

THE ESTIMATOR. For metric m, treat each candidate's true value as drawn from
a population, and its folds as noisy draws around it:

    theta_i ~ N(mu, tau^2)            between-candidate spread
    x_ij | theta_i ~ N(theta_i, sigma^2)   fold noise

The posterior mean is the candidate's own mean pulled toward the population
mean by a factor that depends only on how much evidence it has:

    theta_hat_i = mu + B_i * (xbar_i - mu),  B_i = k_i*tau^2 / (k_i*tau^2 + sigma^2)

B_i is exactly the reliability R(k_i). With the measured single-fold
reliability of 0.373, a 1-fold estimate keeps 37% of its deviation from the
mean and a 40-fold estimate keeps 96%. Nothing here is tuned: mu, sigma^2 and
tau^2 are estimated from the observations themselves, which is the
"empirical" in empirical Bayes.

WHAT IT DOES NOT DO. It cannot rescue a metric we measure badly, and it does
not change which metrics matter. It only stops an unlucky single fold from
outranking a well-measured candidate.
"""
from __future__ import annotations

import statistics

#: Below this many multiply-observed candidates the within-candidate variance
#: cannot be estimated, and shrinking on a guessed sigma^2 would be worse than
#: not shrinking at all.
MINIMUM_REPLICATED = 8


def variance_components(groups: list[list[float]]) -> tuple[float, float, float]:
    """(population mean, within-candidate variance, between-candidate variance).

    `groups` is one list of observations per candidate. Within-candidate
    variance is pooled over the candidates that have at least two folds;
    between-candidate variance is recovered by method of moments from the
    spread of candidate means, which is inflated by sigma^2/k and must have
    that inflation removed.
    """
    usable = [g for g in groups if g]
    if len(usable) < 2:
        raise ValueError("need at least two candidates")
    means = [statistics.mean(g) for g in usable]
    mu = statistics.mean(means)

    replicated = [g for g in usable if len(g) >= 2]
    if len(replicated) < MINIMUM_REPLICATED:
        # No usable estimate of fold noise: report zero within-variance, which
        # makes the shrinkage factor 1 and leaves every estimate untouched.
        return mu, 0.0, statistics.pvariance(means) if len(means) > 1 else 0.0

    ss = sum(sum((x - statistics.mean(g)) ** 2 for x in g) for g in replicated)
    df = sum(len(g) - 1 for g in replicated)
    within = ss / df if df else 0.0

    observed = statistics.pvariance(means)
    harmonic_k = len(usable) / sum(1.0 / len(g) for g in usable)
    between = observed - within / harmonic_k
    return mu, within, max(between, 0.0)


def shrink(mean: float, count: int, mu: float, within: float,
           between: float) -> float:
    """Pull an estimate toward `mu` by how little evidence supports it."""
    if count < 1:
        return mu
    if within <= 0.0 or between <= 0.0:
        return mean          # nothing to shrink toward, or no noise to fight
    weight = (count * between) / (count * between + within)
    return mu + weight * (mean - mu)


def reliability(count: int, within: float, between: float) -> float:
    """The shrinkage weight itself: 0 keeps nothing, 1 keeps the raw mean."""
    if count < 1 or within <= 0.0 or between <= 0.0:
        return 0.0 if count < 1 else 1.0
    return (count * between) / (count * between + within)


def shrunk_metrics(
    observations: dict[str, list[dict[str, float]]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Shrink every metric of every candidate, each metric independently.

    `observations` maps a candidate key to its folds. Metrics are shrunk
    separately because they have different noise: design_iiptm varies far
    more between folds than design_ptm does, so it earns more shrinkage at
    the same depth.
    """
    result: dict[str, dict[str, float]] = {key: {} for key in observations}
    for metric in metrics:
        groups, keys = [], []
        for key, folds in observations.items():
            values = [f[metric] for f in folds
                      if isinstance(f.get(metric), (int, float))]
            if values:
                groups.append(values)
                keys.append(key)
        if len(groups) < 2:
            continue
        mu, within, between = variance_components(groups)
        for key, values in zip(keys, groups):
            result[key][metric] = shrink(
                statistics.mean(values), len(values), mu, within, between)
    return result

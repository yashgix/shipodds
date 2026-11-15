"""Replay gate rules on resampled recorded trials."""

from functools import cache

import numpy as np

from .data import TRIALS
from .gates import (
    NAIVE_PEEK,
    OBRIEN_FLEMING,
    POCOCK,
    ROUNDS,
    RULES,
    always_hold_gate,
    bootstrap_gate,
    bootstrap_weights,
    cmh_gate,
    mean_gate,
    sequential_gate,
)

SEED = 20261114
REPLICATES = 2000
FIXED_TRIALS = (1, 2, 4)


@cache
def weights_for(n):
    """Return the task-bootstrap weights for a suite of n tasks."""
    return bootstrap_weights(n, np.random.default_rng([SEED, 0, n]))


def draw(a, b, rng, replicates):
    """Draw trial indices for both arms and return outcomes and paired costs."""
    n = len(a.tasks)
    ia = rng.integers(0, TRIALS, (replicates, n, ROUNDS))
    ib = rng.integers(0, TRIALS, (replicates, n, ROUNDS))
    tasks = np.arange(n)[None, :, None]
    ya = a.rewards[tasks, ia]
    yb = b.rewards[tasks, ib]
    cost = a.costs[tasks, ia] + b.costs[tasks, ib]
    return ya, yb, cost


def outcomes(ya, yb, cost, weights):
    """Run every gate rule on drawn outcomes, returning decisions, runs and USD."""
    n = ya.shape[1]
    results = {}
    for k in FIXED_TRIALS:
        sa = ya[:, :, :k].sum(axis=2)
        sb = yb[:, :, :k].sum(axis=2)
        usd = cost[:, :, :k].sum(axis=(1, 2))
        runs = np.full(ya.shape[0], 2 * n * k, dtype=np.int64)
        results[f"mean_{k}"] = (mean_gate(sa, sb), runs, usd)
        results[f"bootstrap_{k}"] = (bootstrap_gate(sa, sb, weights), runs, usd)
        results[f"cmh_{k}"] = (cmh_gate(sa, sb, k), runs, usd)

    results["sequential_pocock"] = _sequential(ya, yb, cost, POCOCK)
    results["sequential_obf"] = _sequential(ya, yb, cost, OBRIEN_FLEMING)
    return {rule: results[rule] for rule in RULES}


def baseline_outcomes(ya, yb, cost):
    """Run the v2 reference rules: always hold, and a fixed CMH test checked after every round."""
    replicates = ya.shape[0]
    nothing = np.zeros(replicates, dtype=np.int64)
    return {
        "always_hold": (always_hold_gate(replicates), nothing, nothing.astype(float)),
        "naive_peek": _sequential(ya, yb, cost, NAIVE_PEEK),
    }


def _sequential(ya, yb, cost, bounds):
    decisions, rounds_used = sequential_gate(ya, yb, bounds)
    mask = np.arange(ROUNDS)[None, :] < rounds_used[:, None]
    usd = (cost * mask[:, None, :]).sum(axis=(1, 2))
    return decisions, 2 * ya.shape[1] * rounds_used, usd


def simulate(configs, pair, index, replicates=REPLICATES, baselines=False):
    """Replay every rule (and, if asked, the reference rules) on one pair and summarize decisions, runs and cost."""
    a = configs[pair.incumbent]
    b = configs[pair.candidate]
    if a.tasks != b.tasks:
        raise ValueError(f"{pair.incumbent} and {pair.candidate} have different tasks")
    rng = np.random.default_rng([SEED, 1, index])
    ya, yb, cost = draw(a, b, rng, replicates)
    results = outcomes(ya, yb, cost, weights_for(len(a.tasks)))
    if baselines:
        results |= baseline_outcomes(ya, yb, cost)
    summary = {}
    for rule, (decisions, runs, usd) in results.items():
        summary[rule] = {
            "promote": float(np.mean(decisions == 1)),
            "hold": float(np.mean(decisions == -1)),
            "inconclusive": float(np.mean(decisions == 0)),
            "runs": float(np.mean(runs)),
            "cost_usd": float(np.mean(usd)),
        }
    return summary

"""Replay gate rules on resampled recorded trials."""

from functools import cache

import numpy as np

from .data import TRIALS
from .gates import (
    FUTILITY,
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


def v2_outcomes(ya, yb, cost):
    """Run the v2 rules: O'Brien-Fleming with futility stops, always hold, and a fixed test checked every round."""
    replicates = ya.shape[0]
    nothing = np.zeros(replicates, dtype=np.int64)
    results = {rule: _sequential(ya, yb, cost, OBRIEN_FLEMING, threshold) for rule, threshold in FUTILITY.items()}
    results["always_hold"] = (always_hold_gate(replicates), nothing, nothing.astype(float))
    results["naive_peek"] = _sequential(ya, yb, cost, NAIVE_PEEK)
    return results


def _sequential(ya, yb, cost, bounds, futility=-np.inf):
    decisions, rounds_used = sequential_gate(ya, yb, bounds, futility)
    mask = np.arange(ROUNDS)[None, :] < rounds_used[:, None]
    usd = (cost * mask[:, None, :]).sum(axis=(1, 2))
    return decisions, 2 * ya.shape[1] * rounds_used, usd


def draw_smoothed(a, b, rng, replicates):
    """Draw fresh trials at each task's smoothed pass rate, (successes + 0.5) / (TRIALS + 1), at its mean cost."""
    shape = (replicates, len(a.tasks), ROUNDS)
    ya = (rng.random(shape) < _smoothed_rate(a)[None, :, None]).astype(np.int64)
    yb = (rng.random(shape) < _smoothed_rate(b)[None, :, None]).astype(np.int64)
    cost = np.broadcast_to((a.costs.mean(axis=1) + b.costs.mean(axis=1))[None, :, None], shape)
    return ya, yb, cost


def _smoothed_rate(config):
    return (config.rewards.sum(axis=1) + 0.5) / (TRIALS + 1)


def simulate(configs, pair, index, replicates=REPLICATES, v2=False, smoothed=False):
    """Replay every rule (and, if asked, the v2 rules) on one pair and summarize decisions, runs and cost.

    With smoothed=True, trials come from draw_smoothed on a separate random stream instead of the recorded trials.
    """
    a = configs[pair.incumbent]
    b = configs[pair.candidate]
    if a.tasks != b.tasks:
        raise ValueError(f"{pair.incumbent} and {pair.candidate} have different tasks")
    if smoothed:
        ya, yb, cost = draw_smoothed(a, b, np.random.default_rng([SEED, 2, index]), replicates)
    else:
        ya, yb, cost = draw(a, b, np.random.default_rng([SEED, 1, index]), replicates)
    results = outcomes(ya, yb, cost, weights_for(len(a.tasks)))
    if v2:
        results |= v2_outcomes(ya, yb, cost)
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

"""Run the promotion-gate experiment and write its metrics."""

import json
from pathlib import Path

from . import DATA_DIR
from .data import TRIALS, load
from .gates import BASELINE_RULES, BOOTSTRAP_DRAWS, ROUNDS, RULES, cmh_gate
from .pairs import aa_pairs, delta, real_pairs
from .simulate import REPLICATES, simulate

ERROR_CAP = 0.05
POWER_SLACK = 0.05
V2_NAME = "metrics_v2.json"


def recorded_decision(configs, pair):
    """Return what a full recorded CMH evaluation says about the pair."""
    incumbent = configs[pair.incumbent]
    candidate = configs[pair.candidate]
    sa = incumbent.rewards.sum(axis=1)
    sb = candidate.rewards.sum(axis=1)
    return int(cmh_gate(sa, sb, TRIALS).item())


def full_cost(configs, pair):
    """Return the recorded USD cost of evaluating both arms on all trials."""
    incumbent = configs[pair.incumbent]
    candidate = configs[pair.candidate]
    return float(incumbent.costs.sum() + candidate.costs.sum())


def recommend(metrics):
    """Return the cheapest rule that meets the error caps and power slack."""
    aa_promote = metrics["aa_promote"]
    worse_promote = metrics["worse_promote"]
    better_promote = metrics["better_promote"]
    mean_cost = metrics["mean_cost_usd"]
    kept = [i for i in range(len(aa_promote)) if aa_promote[i] <= ERROR_CAP and worse_promote[i] <= ERROR_CAP]
    if not kept:
        return -1
    best_better = max(better_promote[i] for i in kept)
    eligible = [i for i in kept if better_promote[i] >= best_better - POWER_SLACK]
    return min(eligible, key=lambda i: (mean_cost[i], i))


def _mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _group_rate(summaries, indices, rule, outcome):
    return _mean(summaries[i][rule][outcome] for i in indices)


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def rule_metrics(real_summaries, aa_summaries, worse, better, rules, full_cost_usd):
    """Return per-rule group outcome rates, mean runs and mean cost for the given rules."""
    metrics = {}
    for group, indices, summaries in (
        ("aa", range(len(aa_summaries)), aa_summaries),
        ("worse", worse, real_summaries),
        ("better", better, real_summaries),
    ):
        for outcome in ("promote", "hold", "inconclusive"):
            metrics[f"{group}_{outcome}"] = [_group_rate(summaries, indices, rule, outcome) for rule in rules]
    real = range(len(real_summaries))
    metrics["mean_runs"] = [_mean(real_summaries[i][rule]["runs"] for i in real) for rule in rules]
    metrics["mean_cost_usd"] = [_mean(real_summaries[i][rule]["cost_usd"] for i in real) for rule in rules]
    metrics["cost_share"] = [cost / full_cost_usd for cost in metrics["mean_cost_usd"]]
    return metrics


def distinct_comparisons(pairs):
    """Count comparisons ignoring direction: a pair and its reverse are one comparison."""
    return len({frozenset((pair.incumbent, pair.candidate)) for pair in pairs})


def _write(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rounded(metrics), indent=1, sort_keys=True) + "\n")


def main(data_dir=DATA_DIR, out="results/metrics.json", replicates=REPLICATES):
    """Simulate every real and identical-copy pair and write the v1 metrics and, beside them, the v2 metrics."""
    configs = load(data_dir)
    real = real_pairs(configs)
    copies = aa_pairs(configs)

    real_summaries = [simulate(configs, pair, i, replicates, baselines=True) for i, pair in enumerate(real)]
    aa_summaries = [simulate(configs, pair, len(real) + i, replicates, baselines=True) for i, pair in enumerate(copies)]

    deltas = [delta(configs, pair) for pair in real]
    full_costs = [full_cost(configs, pair) for pair in real]
    recorded = [recorded_decision(configs, pair) for pair in real]
    worse = [i for i, value in enumerate(deltas) if value <= 0]
    better = [i for i, value in enumerate(deltas) if value > 0]

    full_cost_usd = _mean(full_costs)
    metrics = {
        "configs": len(configs),
        "runs_recorded": sum(config.rewards.size for config in configs.values()),
        "pairs": len(real),
        "aa_pairs": len(copies),
        "worse_pairs": len(worse),
        "better_pairs": len(better),
        "replicates": replicates,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "max_rounds": ROUNDS,
        "full_cost_usd": full_cost_usd,
        "recorded_promote": sum(value == 1 for value in recorded),
        "recorded_hold": sum(value == -1 for value in recorded),
        "recorded_inconclusive": sum(value == 0 for value in recorded),
        "pair_delta": deltas,
        "pair_tasks": [len(configs[pair.incumbent].tasks) for pair in real],
        "pair_full_cost_usd": full_costs,
        "pair_recorded": recorded,
        "pair_promote": [real_summaries[i][rule]["promote"] for rule in RULES for i in range(len(real))],
        "pair_hold": [real_summaries[i][rule]["hold"] for rule in RULES for i in range(len(real))],
        "aa_pair_promote": [aa_summaries[i][rule]["promote"] for rule in RULES for i in range(len(copies))],
    }
    metrics |= rule_metrics(real_summaries, aa_summaries, worse, better, RULES, full_cost_usd)
    metrics["recommended_rule"] = recommend(metrics)

    v2 = {"rules": list(BASELINE_RULES), "distinct_comparisons": distinct_comparisons(real)}
    v2 |= rule_metrics(real_summaries, aa_summaries, worse, better, BASELINE_RULES, full_cost_usd)

    path = Path(out)
    _write(path, metrics)
    _write(path.with_name(V2_NAME), v2)

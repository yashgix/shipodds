"""Run the promotion-gate experiment and write its metrics."""

import json
from pathlib import Path

from . import DATA_DIR
from .data import TRIALS, load
from .gates import BOOTSTRAP_DRAWS, ROUNDS, RULES, cmh_gate
from .pairs import aa_pairs, delta, real_pairs
from .robustness import WORLDS, Replay, v2_metrics
from .simulate import REPLICATES, simulate
from .summary import mean, recommend, rule_metrics

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


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rounded(metrics), indent=1, sort_keys=True) + "\n")


def main(data_dir=DATA_DIR, out="results/metrics.json", replicates=REPLICATES):
    """Simulate every real and identical-copy pair and write the v1 metrics and, beside them, the v2 metrics."""
    configs = load(data_dir)
    real = real_pairs(configs)
    copies = aa_pairs(configs)
    replays = {}
    for world in WORLDS:
        smoothed = world == "smoothed"
        replays[world] = (
            [simulate(configs, pair, i, replicates, v2=True, smoothed=smoothed) for i, pair in enumerate(real)],
            [
                simulate(configs, pair, len(real) + i, replicates, v2=True, smoothed=smoothed)
                for i, pair in enumerate(copies)
            ],
        )
    real_summaries, aa_summaries = replays["recorded"]

    deltas = [delta(configs, pair) for pair in real]
    full_costs = [full_cost(configs, pair) for pair in real]
    recorded = [recorded_decision(configs, pair) for pair in real]
    worse = [i for i, value in enumerate(deltas) if value <= 0]
    better = [i for i, value in enumerate(deltas) if value > 0]

    full_cost_usd = mean(full_costs)
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

    aa_costs = [full_cost(configs, pair) for pair in copies]
    v2 = v2_metrics(
        {world: Replay(*replay, deltas, full_costs, aa_costs) for world, replay in replays.items()},
        real,
        copies,
    )

    path = Path(out)
    _write(path, metrics)
    _write(path.with_name(V2_NAME), v2)

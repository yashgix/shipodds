"""Aggregate per-pair simulation summaries into per-rule metrics, and pick a rule."""

ERROR_CAP = 0.05
POWER_SLACK = 0.05


def recommend(metrics, error_cap=ERROR_CAP, power_slack=POWER_SLACK):
    """Return the position of the cheapest rule that meets the error caps and power slack, or -1."""
    aa_promote = metrics["aa_promote"]
    worse_promote = metrics["worse_promote"]
    better_promote = metrics["better_promote"]
    mean_cost = metrics["mean_cost_usd"]
    kept = [i for i in range(len(aa_promote)) if aa_promote[i] <= error_cap and worse_promote[i] <= error_cap]
    if not kept:
        return -1
    best_better = max(better_promote[i] for i in kept)
    eligible = [i for i in kept if better_promote[i] >= best_better - power_slack]
    return min(eligible, key=lambda i: (mean_cost[i], i))


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def group_rate(summaries, indices, rule, outcome):
    return mean(summaries[i][rule][outcome] for i in indices)


def rule_metrics(real_summaries, aa_summaries, worse, better, rules, full_cost_usd):
    """Return per-rule group outcome rates, mean runs and mean cost for the given rules."""
    metrics = {}
    for group, indices, summaries in (
        ("aa", range(len(aa_summaries)), aa_summaries),
        ("worse", worse, real_summaries),
        ("better", better, real_summaries),
    ):
        for outcome in ("promote", "hold", "inconclusive"):
            metrics[f"{group}_{outcome}"] = [group_rate(summaries, indices, rule, outcome) for rule in rules]
    real = range(len(real_summaries))
    metrics["mean_runs"] = [mean(real_summaries[i][rule]["runs"] for i in real) for rule in rules]
    metrics["mean_cost_usd"] = [mean(real_summaries[i][rule]["cost_usd"] for i in real) for rule in rules]
    metrics["cost_share"] = [cost / full_cost_usd for cost in metrics["mean_cost_usd"]]
    return metrics


def distinct_comparisons(pairs):
    """Count comparisons ignoring direction: a pair and its reverse are one comparison."""
    return len({frozenset((pair.incumbent, pair.candidate)) for pair in pairs})

"""Render README results and the one-line demo summary from metrics.json and metrics_v2.json."""

from .data import load
from .gates import FUTILITY, RULES
from .pairs import real_pairs

MODEL_NAMES = {
    "claude-3-7-sonnet-20250219": "claude-3.7-sonnet",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "o4-mini-2025-04-16": "o4-mini",
}
MARKER_START = "<!-- results:start -->"
MARKER_END = "<!-- results:end -->"
DECISIONS = {1: "promote", -1: "hold", 0: "inconclusive"}
PAIR_RULES = ("mean_1", "cmh_4", "sequential_obf")
PAIR_RULE_POSITIONS = tuple(RULES.index(rule) for rule in PAIR_RULES)
TRADEOFF_RULES = (
    "always_hold",
    "mean_1",
    "mean_4",
    "cmh_1",
    "cmh_4",
    "naive_peek",
    "sequential_pocock",
    "sequential_obf",
    *FUTILITY,
)
TRADEOFF_KEYS = (
    "aa_promote",
    "worse_promote",
    "better_promote",
    "better_hold",
    "better_inconclusive",
    "cost_share",
    "aa_cost_share",
)
TRADEOFF_HEADER = (
    "Identical copy promoted",
    "Worse promoted",
    "Better promoted (power)",
    "Better held (false rejection)",
    "Better inconclusive (incumbent kept)",
    "Cost, real pairs",
    "Cost, identical copies",
)
FOLD_KEYS = ("aa_promote", "worse_promote", "better_promote", "better_hold", "cost_share", "aa_cost_share")
CHECK_RULES = ("mean_1", "cmh_1", "cmh_4", "sequential_obf")


def short_model(model):
    """Return a compact model name for a table cell."""
    return MODEL_NAMES.get(model, model)


def short_name(key):
    """Return a compact config label, showing the mode only when it is not default."""
    model, domain, mode = key
    label = f"{short_model(model)} {domain}"
    return label if mode == "default" else f"{label} ({mode})"


def pct(value):
    """Format a fraction as a percentage with one decimal."""
    return f"{100.0 * value:.1f}%"


def _points(value):
    points = round(100.0 * value, 1)
    return "0.0" if points == 0 else f"{points:+.1f}"


def _decision(value):
    return DECISIONS.get(int(value), "inconclusive")


def summary_line(metrics):
    """Return the one-line demo summary."""
    naive = metrics["aa_promote"][0]
    chosen = metrics["recommended_rule"]
    if not 0 <= chosen < len(RULES):
        return f"mean_1 promotes an identical copy {pct(naive)} of the time; no rule meets the error caps."
    return (
        f"mean_1 promotes an identical copy {pct(naive)} of the time; "
        f"the v1 pick ({RULES[chosen]}) {pct(metrics['aa_promote'][chosen])}, "
        f"at {pct(metrics['cost_share'][chosen])} of a full evaluation's cost."
    )


def results_paragraph(metrics):
    """Return the short paragraph that opens the generated results block."""
    return (
        f"Across {metrics['pairs']} candidate/incumbent pairs ({metrics['configs']} configs, "
        f"{metrics['runs_recorded']:,} recorded runs, and {metrics['aa_pairs']} identical-copy pairs), "
        f"each gate was replayed {metrics['replicates']:,} times. A full {metrics['max_rounds']}-trial "
        f"evaluation of both arms costs ${metrics['full_cost_usd']:,.2f} on average. The full recorded "
        f"evaluation promotes {metrics['recorded_promote']} pairs, holds {metrics['recorded_hold']}, "
        f"and cannot decide {metrics['recorded_inconclusive']}."
    )


def _real_inconclusive(metrics, position):
    worse = metrics["worse_pairs"]
    better = metrics["better_pairs"]
    total = worse + better
    if total == 0:
        return 0.0
    weighted = metrics["worse_inconclusive"][position] * worse + metrics["better_inconclusive"][position] * better
    return weighted / total


def table(header, rows):
    """Return a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def rule_table(metrics):
    """Return the per-rule markdown table."""
    header = (
        "Rule",
        "Identical copy promoted",
        "Worse promoted",
        "Better promoted",
        "Inconclusive",
        "Runs per decision",
        "Cost per decision",
        "Share of full evaluation",
    )
    rows = []
    chosen = metrics["recommended_rule"]
    for position, rule in enumerate(RULES):
        label = f"**{rule}** (recommended)" if position == chosen else rule
        row = (
            label,
            pct(metrics["aa_promote"][position]),
            pct(metrics["worse_promote"][position]),
            pct(metrics["better_promote"][position]),
            pct(_real_inconclusive(metrics, position)),
            f"{metrics['mean_runs'][position]:,.0f}",
            f"${metrics['mean_cost_usd'][position]:,.2f}",
            pct(metrics["cost_share"][position]),
        )
        rows.append(row)
    return table(header, rows)


def pair_table(metrics, pairs=None):
    """Return the per-pair markdown table, sorted by true difference (largest first)."""
    if pairs is None:
        pairs = real_pairs(load())
    count = len(metrics["pair_delta"])
    if len(pairs) < count:
        raise ValueError("not enough pair labels for the metrics")
    header = (
        "Incumbent",
        "Candidate",
        "Tasks",
        "True difference (pts)",
        "Full recorded evaluation",
        *PAIR_RULES,
    )
    rows = []
    for position in range(count):
        pair = pairs[position]
        rates = tuple(
            pct(metrics["pair_promote"][rule_position * count + position]) for rule_position in PAIR_RULE_POSITIONS
        )
        rows.append(
            (
                metrics["pair_delta"][position],
                position,
                (
                    short_name(pair.incumbent),
                    short_name(pair.candidate),
                    str(metrics["pair_tasks"][position]),
                    _points(metrics["pair_delta"][position]),
                    _decision(metrics["pair_recorded"][position]),
                    *rates,
                ),
            )
        )
    rows.sort(key=lambda row: (-row[0], row[1]))
    return table(header, [row[2] for row in rows])


def _span(values):
    low, high = pct(min(values)), pct(max(values))
    return low if low == high else f"{low} to {high}"


def _at(metrics, v2, rule, key):
    return metrics[key][v2["rules"].index(rule)]


def headline(v2):
    """Return the generated headline: the held-out check first, then the two in-sample effects."""
    folds = v2["folds"]
    picks = sorted({fold["pick"] for fold in folds})

    def held(rule, key):
        return _span([_at(fold, v2, rule if rule else fold["pick"], key) for fold in folds])

    loss = max(_at(f, v2, "cmh_4", "better_promote") - _at(f, v2, f["pick"], "better_promote") for f in folds)
    return (
        "**Held out by domain (exploratory).** The selection procedure and the futility thresholds were designed "
        "after looking at all domains, so this is a post-selection check, not an independent validation. Picking "
        "the rule on two domains and scoring it on the third, the v1 selection procedure picks "
        f"{' or '.join(picks)} in each of the {len(folds)} folds. On the held-out domain the pick "
        f"promotes an identical copy {held(None, 'aa_promote')} of the time (cmh_4: {held('cmh_4', 'aa_promote')}). "
        f"It promotes {held(None, 'better_promote')} of better candidates against {held('cmh_4', 'better_promote')} "
        f"for the fixed four-trial cmh_4, at most {100 * loss:.1f} points fewer. It costs {held(None, 'cost_share')} "
        f"of a full evaluation on real pairs and {held(None, 'aa_cost_share')} on identical copies. The one-shot "
        f"cmh_1, at a quarter of the cost, promotes only {held('cmh_1', 'better_promote')} of better candidates.\n\n"
        + effects(v2)
    )


def effects(v2):
    """Return the in-sample paragraph that separates the two effects and the peeking check."""

    def at(rule, key):
        return pct(_at(v2, v2, rule, key))

    return (
        "**Two separate effects** (all pairs, in sample). A proper test instead of the mean gate is what lowers "
        f"false promotions. With one trial per task and the same cost, mean_1 promotes an identical copy "
        f"{at('mean_1', 'aa_promote')} of the time and cmh_1 {at('cmh_1', 'aa_promote')}. A sequential test instead "
        f"of a fixed one is what lowers cost. sequential_obf promotes {at('sequential_obf', 'better_promote')} of "
        f"better candidates against {at('cmh_4', 'better_promote')} for cmh_4, at {at('sequential_obf', 'cost_share')} "
        f"of the cost, but {at('sequential_obf', 'aa_cost_share')} on identical copies, where it rarely stops early. "
        "Futility stops target that case. Checking the fixed test after every round "
        f"(naive_peek) promotes an identical copy {at('naive_peek', 'aa_promote')} of the time. Its power "
        f"({at('naive_peek', 'better_promote')}) and cost ({at('naive_peek', 'cost_share')}) come from that looser "
        "error rate."
    )


def caveat(metrics, v2):
    """Return the paragraph that says how far the numbers can be trusted."""
    folds = v2["folds"]
    sizes = ", ".join(f"{f['held_out']} {f['distinct_comparisons']} and {f['aa_pairs']}" for f in folds)
    text = (
        f"**How small this is.** The {metrics['pairs']} pairs are only {v2['distinct_comparisons']} distinct "
        "comparisons, each counted in both directions, and some configs appear in several comparisons. Each held-out "
        f"domain has few distinct comparisons and identical-copy pairs ({sizes}). There are only {len(folds)} folds, "
        "and no interval is given. The v1 recommended rule was picked on all the data, with caps set after a "
        "prototype run on the same data."
    )
    largest = list(FUTILITY)[-1]
    if all(fold["pick"] == largest for fold in folds):
        text += f" Every fold picked the largest futility threshold tried ({largest}). Larger thresholds are untested."
    return text


def tradeoff_table(v2):
    """Return the side-by-side table of false promotions, power, false rejections and cost."""
    rows = [(rule, *(pct(_at(v2, v2, rule, key)) for key in TRADEOFF_KEYS)) for rule in TRADEOFF_RULES]
    return table(("Rule", *TRADEOFF_HEADER), rows)


def fold_table(v2):
    """Return the held-out table: each fold's pick and the fixed tests, scored on the held-out domain."""
    header = ("Held out", "Rule", *(TRADEOFF_HEADER[TRADEOFF_KEYS.index(key)] for key in FOLD_KEYS))
    rows = []
    for fold in v2["folds"]:
        for rule in (fold["pick"], "sequential_obf", "cmh_4", "cmh_1"):
            label = f"{rule} (pick)" if rule == fold["pick"] else rule
            rows.append((fold["held_out"], label, *(pct(_at(fold, v2, rule, key)) for key in FOLD_KEYS)))
    return table(header, rows)


def difference_table(v2):
    """Return power by size of the true difference, for better pairs."""
    bins = v2["power_by_difference"]
    edges = [round(100 * edge) for edge in bins["edges"]]
    header = [
        f"+{low} to {high} pts ({n} pairs)" for low, high, n in zip(edges, edges[1:], bins["pairs"], strict=False)
    ]
    rules = (*CHECK_RULES, v2["pick"])
    rows = [(rule, *map(pct, bins["better_promote"][v2["rules"].index(rule)])) for rule in rules]
    return table(("Rule", *header), rows)


def smoothed_table(v2):
    """Return recorded against smoothed-noise results for the main rules."""
    smoothed = v2["smoothed"]
    keys = ("aa_promote", "worse_promote", "better_promote", "cost_share")
    header = ("Rule", *(f"{TRADEOFF_HEADER[TRADEOFF_KEYS.index(key)]}, recorded / smoothed" for key in keys))
    rows = [
        (rule, *(f"{pct(_at(v2, v2, rule, key))} / {pct(_at(smoothed, v2, rule, key))}" for key in keys))
        for rule in (*CHECK_RULES, v2["pick"])
    ]
    return table(header, rows) + f"\n\nThe smoothed world picks {smoothed['pick']}."


def grid_table(v2):
    """Return the pick for each error cap (rows) and power slack (columns)."""
    grid = v2["pick_grid"]
    header = ("Error cap", *(f"slack {round(100 * slack)} pts" for slack in grid["power_slack"]))
    rows = [
        (pct(cap), *(rule or "none" for rule in picks))
        for cap, picks in zip(grid["error_cap"], grid["pick"], strict=True)
    ]
    return table(header, rows)


def results_block(metrics, v2, pairs=None):
    """Return the complete text that belongs between the README markers."""
    parts = (
        headline(v2),
        caveat(metrics, v2),
        "**All pairs, every rule that matters for the headline.**",
        tradeoff_table(v2),
        "**Held out by domain.** Picked on the other two domains, scored on this one.",
        fold_table(v2),
        "**Power by true difference** (better pairs, all domains).",
        difference_table(v2),
        "**Smoothed-noise world.** Each task's pass rate is (successes + 0.5) / 5 and trials are drawn fresh.",
        smoothed_table(v2),
        "**How the pick moves with the caps** (all domains).",
        grid_table(v2),
        "**v1 results.**",
        results_paragraph(metrics),
        rule_table(metrics),
        pair_table(metrics, pairs),
    )
    return "\n\n".join(parts)


def replace_results(text, block):
    """Replace the text between the results markers, keeping the marker lines."""
    start = text.find(MARKER_START)
    if start == -1:
        raise ValueError(f"missing {MARKER_START!r}")
    end = text.find(MARKER_END, start + len(MARKER_START))
    if end == -1:
        raise ValueError(f"missing {MARKER_END!r} after {MARKER_START!r}")
    return f"{text[: start + len(MARKER_START)]}\n{block.strip()}\n{text[end:]}"


def render_readme(text, metrics, v2, pairs=None):
    """Return the README with its results block rendered from the v1 and v2 metrics."""
    return replace_results(text, results_block(metrics, v2, pairs))

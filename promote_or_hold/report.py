"""Render README results and the one-line demo summary from metrics.json."""

from .data import load
from .gates import RULES
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
        f"the recommended rule ({RULES[chosen]}) {pct(metrics['aa_promote'][chosen])}, "
        f"at {pct(metrics['cost_share'][chosen])} of a full evaluation's cost."
    )


def results_paragraph(metrics):
    """Return the short paragraph that opens the generated results block."""
    return (
        f"Across {metrics['pairs']} candidate/incumbent pairs ({metrics['configs']} configs, "
        f"{metrics['runs_recorded']:,} recorded runs; {metrics['aa_pairs']} identical-copy pairs), "
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
    weighted = (
        metrics["worse_inconclusive"][position] * worse
        + metrics["better_inconclusive"][position] * better
    )
    return weighted / total


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
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
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
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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
            pct(metrics["pair_promote"][rule_position * count + position])
            for rule_position in PAIR_RULE_POSITIONS
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
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row[2]) + " |" for row in rows)
    return "\n".join(lines)


def results_block(metrics, pairs=None):
    """Return the complete text that belongs between the README markers."""
    return "\n\n".join((results_paragraph(metrics), rule_table(metrics), pair_table(metrics, pairs)))


def replace_results(text, block):
    """Replace the text between the results markers, keeping the marker lines."""
    start = text.find(MARKER_START)
    if start == -1:
        raise ValueError(f"missing {MARKER_START!r}")
    end = text.find(MARKER_END, start + len(MARKER_START))
    if end == -1:
        raise ValueError(f"missing {MARKER_END!r} after {MARKER_START!r}")
    return f"{text[: start + len(MARKER_START)]}\n{block.strip()}\n{text[end:]}"


def render_readme(text, metrics, pairs=None):
    """Return the README with its results block rendered from metrics."""
    return replace_results(text, results_block(metrics, pairs))

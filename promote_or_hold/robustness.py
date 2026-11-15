"""v2 checks: selection held out by domain, a smoothed-noise world, power by true difference, and the pick grid."""

from dataclasses import dataclass

from .gates import ALL_RULES, SELECTABLE
from .pairs import DOMAINS
from .summary import ERROR_CAP, POWER_SLACK, distinct_comparisons, mean, recommend, rule_metrics

WORLDS = ("recorded", "smoothed")
DELTA_BINS = (0.0, 0.03, 0.06, 0.10, 0.20)
ERROR_CAPS = (0.025, 0.05, 0.10)
POWER_SLACKS = (0.0, 0.02, 0.05, 0.10, 0.20)
SELECTION_KEYS = ("aa_promote", "worse_promote", "better_promote", "mean_cost_usd")


@dataclass(frozen=True)
class Replay:
    """Per-pair summaries of every rule in one simulated world, with the facts needed to score them."""

    real: list
    aa: list
    deltas: list
    real_costs: list
    aa_costs: list

    def subset(self, real_positions, aa_positions):
        """Return the replay restricted to the given real and identical-copy pairs."""
        return Replay(
            [self.real[i] for i in real_positions],
            [self.aa[i] for i in aa_positions],
            [self.deltas[i] for i in real_positions],
            [self.real_costs[i] for i in real_positions],
            [self.aa_costs[i] for i in aa_positions],
        )

    def metrics(self):
        """Return per-rule metrics over ALL_RULES, plus the cost share on identical copies."""
        worse = [i for i, value in enumerate(self.deltas) if value <= 0]
        better = [i for i, value in enumerate(self.deltas) if value > 0]
        metrics = rule_metrics(self.real, self.aa, worse, better, ALL_RULES, mean(self.real_costs))
        aa_full = mean(self.aa_costs)
        metrics["aa_cost_share"] = [mean(s[rule]["cost_usd"] for s in self.aa) / aa_full for rule in ALL_RULES]
        return metrics


def domain(key):
    """Return the task domain of a config key, counting telecom-workflow as telecom."""
    return key[1].removesuffix("-workflow")


def pick(metrics, error_cap=ERROR_CAP, power_slack=POWER_SLACK):
    """Return the rule the v1 selection procedure picks among SELECTABLE, or None."""
    selectable = {key: metrics[key][: len(SELECTABLE)] for key in SELECTION_KEYS}
    position = recommend(selectable, error_cap, power_slack)
    return SELECTABLE[position] if position >= 0 else None


def folds(replay, real, copies):
    """Pick a rule on two domains and score every rule on the third, once per held-out domain."""
    results = []
    for held_out in DOMAINS:
        test_real = [i for i, pair in enumerate(real) if domain(pair.incumbent) == held_out]
        test_aa = [i for i, pair in enumerate(copies) if domain(pair.incumbent) == held_out]
        train_real = [i for i in range(len(real)) if i not in test_real]
        train_aa = [i for i in range(len(copies)) if i not in test_aa]
        results.append(
            {
                "held_out": held_out,
                "pairs": len(test_real),
                "distinct_comparisons": distinct_comparisons([real[i] for i in test_real]),
                "aa_pairs": len(test_aa),
                "pick": pick(replay.subset(train_real, train_aa).metrics()),
                **replay.subset(test_real, test_aa).metrics(),
            }
        )
    return results


def power_by_difference(replay):
    """Return the better-promoted rate of every rule for true differences binned by DELTA_BINS."""
    bins = [
        [i for i, value in enumerate(replay.deltas) if low < value <= high]
        for low, high in zip(DELTA_BINS, DELTA_BINS[1:], strict=False)
    ]
    return {
        "edges": list(DELTA_BINS),
        "pairs": [len(members) for members in bins],
        "better_promote": [
            [mean(replay.real[i][rule]["promote"] for i in members) for members in bins] for rule in ALL_RULES
        ],
    }


def pick_grid(metrics):
    """Return the pick for every combination of error cap and power slack."""
    return {
        "error_cap": list(ERROR_CAPS),
        "power_slack": list(POWER_SLACKS),
        "pick": [[pick(metrics, cap, slack) for slack in POWER_SLACKS] for cap in ERROR_CAPS],
    }


def v2_metrics(replays, real, copies):
    """Return everything SPEC-v2 adds, from the recorded and smoothed replays."""
    recorded = replays["recorded"].metrics()
    smoothed = replays["smoothed"].metrics()
    return {
        "rules": list(ALL_RULES),
        "selectable": list(SELECTABLE),
        "distinct_comparisons": distinct_comparisons(real),
        **recorded,
        "pick": pick(recorded),
        "folds": folds(replays["recorded"], real, copies),
        "smoothed": {**smoothed, "pick": pick(smoothed)},
        "power_by_difference": power_by_difference(replays["recorded"]),
        "pick_grid": pick_grid(recorded),
    }

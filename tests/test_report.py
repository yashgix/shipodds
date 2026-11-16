"""Tests for the generated results block and the demo summary line."""

import unittest
from pathlib import Path

from shipodds.gates import ALL_RULES, FUTILITY, RULES
from shipodds.pairs import DOMAINS, Pair
from shipodds.report import (
    MARKER_END,
    MARKER_START,
    TRADEOFF_RULES,
    caveat,
    difference_table,
    fold_table,
    grid_table,
    headline,
    pair_table,
    pct,
    replace_results,
    results_paragraph,
    rule_table,
    smoothed_table,
    summary_line,
    tradeoff_table,
)


def make_pairs():
    return [
        Pair(
            "model",
            ("gpt-4.1-2025-04-14", "airline", "default"),
            ("gpt-4.1-2025-04-14", "retail", "default"),
        ),
        Pair(
            "model",
            ("o4-mini-2025-04-16", "telecom", "default"),
            ("o4-mini-2025-04-16", "telecom-workflow", "no-user"),
        ),
        Pair(
            "model",
            ("claude-3-7-sonnet-20250219", "airline", "default"),
            ("claude-3-7-sonnet-20250219", "airline", "op"),
        ),
    ]


def make_metrics():
    pairs = 3
    rules = len(RULES)
    metrics = {
        "configs": 3,
        "runs_recorded": 12,
        "pairs": pairs,
        "aa_pairs": 3,
        "worse_pairs": 2,
        "better_pairs": 1,
        "replicates": 5,
        "bootstrap_draws": 1000,
        "max_rounds": 4,
        "full_cost_usd": 12.34,
        "recorded_promote": 1,
        "recorded_hold": 1,
        "recorded_inconclusive": 1,
        "recommended_rule": 4,
        "pair_delta": [0.1, -0.2, 0.0],
        "pair_tasks": [50, 114, 114],
        "pair_full_cost_usd": [10.0, 20.0, 30.0],
        "pair_recorded": [1, -1, 0],
        "pair_promote": [0.1] * (rules * pairs),
        "pair_hold": [0.8] * (rules * pairs),
        "aa_pair_promote": [0.1] * (rules * 3),
    }
    for key in (
        "aa_promote",
        "aa_hold",
        "aa_inconclusive",
        "worse_promote",
        "worse_hold",
        "worse_inconclusive",
        "better_promote",
        "better_hold",
        "better_inconclusive",
    ):
        metrics[key] = [0.1] * rules
    metrics["aa_promote"][0] = 0.439
    metrics["aa_promote"][4] = 0.024
    metrics["worse_promote"][4] = 0.001
    metrics["better_promote"][4] = 0.56
    metrics["worse_inconclusive"][4] = 0.4
    metrics["better_inconclusive"][4] = 0.1
    metrics["mean_runs"] = [100.0] * rules
    metrics["mean_cost_usd"] = [1.0] * rules
    metrics["mean_cost_usd"][4] = 0.8
    metrics["cost_share"] = [0.1] * rules
    metrics["cost_share"][4] = 0.7
    return metrics


PER_RULE_KEYS = (
    "aa_promote",
    "worse_promote",
    "better_promote",
    "better_hold",
    "better_inconclusive",
    "cost_share",
    "aa_cost_share",
)


def rule_lists(value):
    return {key: [value] * len(ALL_RULES) for key in PER_RULE_KEYS}


def set_rate(metrics, rule, key, value):
    metrics[key][ALL_RULES.index(rule)] = value


def make_v2():
    largest = list(FUTILITY)[-1]
    v2 = {"rules": list(ALL_RULES), "distinct_comparisons": 26, "pick": largest, **rule_lists(0.1)}
    v2["folds"] = []
    for held_out, power in zip(DOMAINS, (0.3, 0.6, 0.8), strict=True):
        fold = {"held_out": held_out, "pairs": 12, "distinct_comparisons": 6, "aa_pairs": 4, "pick": largest}
        fold |= rule_lists(0.5)
        set_rate(fold, "cmh_4", "better_promote", power + 0.04)
        set_rate(fold, largest, "better_promote", power)
        v2["folds"].append(fold)
    v2["smoothed"] = {**rule_lists(0.3), "pick": "cmh_4"}
    v2["power_by_difference"] = {
        "edges": [0.0, 0.03, 0.06, 0.1, 0.2],
        "pairs": [5, 5, 10, 6],
        "better_promote": [[0.5] * 4 for _ in ALL_RULES],
    }
    v2["pick_grid"] = {"error_cap": [0.05], "power_slack": [0.0, 0.05], "pick": [["cmh_4", None]]}
    set_rate(v2, "mean_1", "aa_promote", 0.439)
    set_rate(v2, "cmh_1", "aa_promote", 0.023)
    set_rate(v2, "naive_peek", "aa_promote", 0.062)
    return v2


class V2TextTests(unittest.TestCase):
    def setUp(self):
        self.metrics = make_metrics()
        self.v2 = make_v2()

    def test_headline_reports_the_held_out_folds_first(self):
        text = headline(self.v2)
        self.assertTrue(text.startswith("**Held out by domain (exploratory).**"))
        self.assertIn("post-selection check, not an independent validation", text)
        self.assertIn(f"picks {list(FUTILITY)[-1]} in each of the 3 folds", text)
        self.assertIn("promotes 30.0% to 80.0% of better candidates against 34.0% to 84.0%", text)
        self.assertIn("at most 4.0 points fewer", text)

    def test_headline_names_both_effects_and_the_peeking_rate(self):
        text = headline(self.v2)
        for fragment in ("mean_1 promotes an identical copy 43.9%", "cmh_1 2.3%", "promotes an identical copy 6.2%"):
            self.assertIn(fragment, text)

    def test_caveat_counts_and_flags_the_grid_edge(self):
        text = caveat(self.metrics, self.v2)
        self.assertIn("3 pairs are only 26 distinct comparisons", text)
        self.assertIn("airline 6 and 4", text)
        self.assertIn("largest futility threshold tried (obf_futility_2.0). Larger thresholds are untested.", text)
        self.v2["folds"][0]["pick"] = "cmh_4"
        self.assertNotIn("largest futility threshold tried", caveat(self.metrics, self.v2))

    def test_tradeoff_table_rows(self):
        rows = [line for line in tradeoff_table(self.v2).splitlines() if line.startswith("| ")][1:]
        self.assertEqual([row.split("|")[1].strip() for row in rows], list(TRADEOFF_RULES))
        self.assertTrue(rows[0].startswith("| always_hold | 10.0% |"))

    def test_fold_table_marks_each_pick(self):
        rows = [line for line in fold_table(self.v2).splitlines() if line.startswith("| ")][1:]
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum("(pick)" in row for row in rows), 3)

    def test_other_tables(self):
        self.assertIn("+0 to 3 pts (5 pairs)", difference_table(self.v2))
        self.assertIn("10.0% / 30.0%", smoothed_table(self.v2))
        self.assertIn("The smoothed world picks cmh_4.", smoothed_table(self.v2))
        self.assertIn("| 5.0% | cmh_4 | none |", grid_table(self.v2))


class PercentTests(unittest.TestCase):
    def test_one_decimal(self):
        self.assertEqual(pct(0.439), "43.9%")
        self.assertEqual(pct(1.0), "100.0%")
        self.assertEqual(pct(0.0), "0.0%")


class TextTests(unittest.TestCase):
    def setUp(self):
        self.metrics = make_metrics()

    def test_paragraph_has_counts_and_recorded_decisions(self):
        text = results_paragraph(self.metrics)
        for fragment in (
            "Across 3 candidate/incumbent pairs",
            "3 configs",
            "12 recorded runs",
            "3 identical-copy pairs",
            "replayed 5 times",
            "$12.34",
            "promotes 1 pairs",
            "holds 1",
            "cannot decide 1",
        ):
            self.assertIn(fragment, text)

    def test_rule_table_marks_recommended_and_has_a_row_per_rule(self):
        table = rule_table(self.metrics)
        self.assertIn("**bootstrap_2** (recommended)", table)
        rows = [line for line in table.splitlines() if line.startswith("|")]
        self.assertEqual(len(rows), len(RULES) + 2)
        for fragment in ("43.9%", "2.4%", "0.1%", "56.0%", "30.0%", "70.0%"):
            self.assertIn(fragment, table)
        recommended_row = next(line for line in rows if "**bootstrap_2**" in line)
        self.assertIn("30.0%", recommended_row)

    def test_pair_table_sorted_by_true_difference(self):
        table = pair_table(self.metrics, make_pairs())
        rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
        data = rows[1:]
        expected = [
            "gpt-4.1 retail",
            "claude-3.7-sonnet airline (op)",
            "o4-mini telecom-workflow (no-user)",
        ]
        self.assertEqual([line.split("|")[2].strip() for line in data], expected)
        self.assertEqual([line.split("|")[4].strip() for line in data], ["+10.0", "0.0", "-20.0"])
        self.assertEqual(
            [line.split("|")[5].strip() for line in data],
            ["promote", "inconclusive", "hold"],
        )


class MarkerTests(unittest.TestCase):
    def test_replacement_keeps_markers_and_outside_text(self):
        text = f"before\n{MARKER_START}\nold\n{MARKER_END}\nafter\n"
        self.assertEqual(
            replace_results(text, "new"),
            f"before\n{MARKER_START}\nnew\n{MARKER_END}\nafter\n",
        )

    def test_missing_markers_raise(self):
        with self.assertRaises(ValueError):
            replace_results("no markers", "new")
        with self.assertRaises(ValueError):
            replace_results(f"{MARKER_START}\n", "new")


class ReadmeMarkerTests(unittest.TestCase):
    def test_repo_readme_has_results_markers(self):
        readme = Path(__file__).resolve().parent.parent / "README.md"
        if not readme.is_file():
            self.skipTest("README.md is written in a later build step")
        text = readme.read_text()
        self.assertIn(MARKER_START, text)
        self.assertIn(MARKER_END, text)


class SummaryTests(unittest.TestCase):
    def test_summary_line(self):
        line = summary_line(make_metrics())
        self.assertIn("mean_1 promotes an identical copy 43.9%", line)
        self.assertIn("bootstrap_2", line)
        self.assertIn("2.4%", line)
        self.assertIn("70.0%", line)

    def test_no_rule_message(self):
        metrics = make_metrics()
        metrics["recommended_rule"] = -1
        self.assertIn("no rule meets the error caps", summary_line(metrics))


if __name__ == "__main__":
    unittest.main()

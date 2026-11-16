"""Tests for the v2 checks: held-out folds, pick, bins and the pick grid."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shipodds.gates import ALL_RULES, SELECTABLE
from shipodds.pairs import Pair
from shipodds.robustness import (
    DELTA_BINS,
    ERROR_CAPS,
    POWER_SLACKS,
    Replay,
    domain,
    folds,
    pick,
    pick_grid,
    power_by_difference,
)
from shipodds.run import V2_NAME, main


def summary(promote=0.5, cost=1.0):
    return {"promote": promote, "hold": 0.0, "inconclusive": 1.0 - promote, "runs": 1.0, "cost_usd": cost}


def uniform(**values):
    return {rule: summary(**values) for rule in ALL_RULES}


def selection_metrics(cheapest):
    """Metrics where every rule meets the caps with equal power and `cheapest` costs least."""
    n = len(ALL_RULES)
    metrics = {"aa_promote": [0.01] * n, "worse_promote": [0.0] * n, "better_promote": [0.7] * n}
    metrics["mean_cost_usd"] = [10.0] * n
    metrics["mean_cost_usd"][ALL_RULES.index(cheapest)] = 1.0
    return metrics


class DomainTests(unittest.TestCase):
    def test_workflow_counts_as_telecom(self):
        self.assertEqual(domain(("m", "telecom-workflow", "op")), "telecom")
        self.assertEqual(domain(("m", "airline", "default")), "airline")


class ReplayTests(unittest.TestCase):
    def test_metrics_and_subset(self):
        replay = Replay(
            [uniform(promote=1.0, cost=5.0), uniform(promote=0.0, cost=15.0)],
            [uniform(promote=0.2, cost=2.0)],
            [0.1, -0.1],
            [10.0, 30.0],
            [8.0],
        )
        metrics = replay.metrics()
        self.assertEqual(metrics["better_promote"][0], 1.0)
        self.assertEqual(metrics["worse_promote"][0], 0.0)
        self.assertAlmostEqual(metrics["cost_share"][0], 10.0 / 20.0)
        self.assertAlmostEqual(metrics["aa_cost_share"][0], 2.0 / 8.0)
        part = replay.subset([1], [0])
        self.assertEqual((part.deltas, part.real_costs), ([-0.1], [30.0]))


class PickTests(unittest.TestCase):
    def test_picks_cheapest_selectable(self):
        self.assertEqual(pick(selection_metrics("obf_futility_0.5")), "obf_futility_0.5")

    def test_reference_rules_are_never_picked(self):
        self.assertIn(pick(selection_metrics("naive_peek")), SELECTABLE)
        self.assertNotEqual(pick(selection_metrics("always_hold")), "always_hold")

    def test_none_when_no_rule_meets_the_caps(self):
        metrics = selection_metrics("cmh_4")
        metrics["aa_promote"] = [0.5] * len(ALL_RULES)
        self.assertIsNone(pick(metrics))

    def test_grid_shape(self):
        grid = pick_grid(selection_metrics("cmh_2"))
        self.assertEqual(len(grid["pick"]), len(ERROR_CAPS))
        self.assertEqual({len(row) for row in grid["pick"]}, {len(POWER_SLACKS)})
        self.assertEqual({rule for row in grid["pick"] for rule in row}, {"cmh_2"})


class FoldTests(unittest.TestCase):
    def test_pick_never_sees_the_held_out_domain(self):
        # Each domain's pairs carry a distinct cost, so the picking metrics reveal which pairs they came from.
        costs = {"airline": 1.0, "retail": 10.0, "telecom": 100.0}
        keys = [("m", d, "default") for d in ("airline", "retail", "telecom", "telecom-workflow")]
        real = [Pair("model", key, key) for key in keys]
        copies = [Pair("aa", key, key) for key in keys]
        replay = Replay(
            [uniform(cost=costs[domain(key)]) for key in keys],
            [uniform(cost=costs[domain(key)]) for key in keys],
            [0.1, -0.1, 0.1, -0.1],
            [1.0] * 4,
            [1.0] * 4,
        )
        with mock.patch("shipodds.robustness.pick", side_effect=lambda m: m["mean_cost_usd"][0]):
            results = folds(replay, real, copies)
        self.assertEqual([f["held_out"] for f in results], ["airline", "retail", "telecom"])
        self.assertEqual([f["pick"] for f in results], [70.0, 67.0, 5.5])
        self.assertEqual([f["pairs"] for f in results], [1, 1, 2])
        self.assertEqual([f["mean_cost_usd"][0] for f in results], [1.0, 10.0, 100.0])


class BinTests(unittest.TestCase):
    def test_better_pairs_are_binned_by_true_difference(self):
        replay = Replay(
            [uniform(promote=p) for p in (0.1, 0.4, 0.9, 0.7)],
            [],
            [0.01, 0.05, -0.02, 0.15],
            [1.0] * 4,
            [],
        )
        bins = power_by_difference(replay)
        self.assertEqual(bins["edges"], list(DELTA_BINS))
        self.assertEqual(bins["pairs"], [1, 1, 0, 1])
        self.assertEqual(bins["better_promote"][0], [0.1, 0.4, 0.0, 0.7])


class RealDataTests(unittest.TestCase):
    def test_v2_metrics_on_the_recorded_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            main(out=Path(folder) / "metrics.json", replicates=20)
            v2 = json.loads((Path(folder) / V2_NAME).read_text())
        self.assertEqual(v2["rules"], list(ALL_RULES))
        self.assertEqual(v2["distinct_comparisons"], 26)
        sizes = [(f["held_out"], f["pairs"], f["distinct_comparisons"], f["aa_pairs"]) for f in v2["folds"]]
        self.assertEqual(sizes, [("airline", 12, 6, 4), ("retail", 12, 6, 4), ("telecom", 28, 14, 18)])
        self.assertTrue(all(f["pick"] in SELECTABLE for f in v2["folds"]))
        self.assertEqual(sum(v2["power_by_difference"]["pairs"]), 26)
        self.assertEqual(len(v2["smoothed"]["aa_promote"]), len(ALL_RULES))

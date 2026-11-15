"""Tests for the experiment runner and its recommendation rule."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from promote_or_hold import DATA_DIR
from promote_or_hold.data import TRIALS, Config
from promote_or_hold.gates import RULES
from promote_or_hold.pairs import Pair
from promote_or_hold.run import (
    ERROR_CAP,
    POWER_SLACK,
    full_cost,
    main,
    recommend,
    recorded_decision,
)


def make_config(model="m", rewards=None, costs=None):
    if rewards is None and costs is None:
        rewards = np.zeros((2, TRIALS), dtype=np.int64)
        costs = np.ones((2, TRIALS), dtype=float)
    elif rewards is None:
        costs = np.asarray(costs, dtype=float)
        rewards = np.zeros(costs.shape, dtype=np.int64)
    elif costs is None:
        rewards = np.asarray(rewards, dtype=np.int64)
        costs = np.ones(rewards.shape, dtype=float)
    rewards = np.asarray(rewards, dtype=np.int64)
    costs = np.asarray(costs, dtype=float)
    tasks = tuple(f"t{i}" for i in range(rewards.shape[0]))
    return Config(model, "d", "default", tasks, rewards, costs)


class RecordedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.low = make_config("low", rewards=np.zeros((2, TRIALS), dtype=np.int64))
        self.high = make_config("high", rewards=np.ones((2, TRIALS), dtype=np.int64))

    def test_promotes_clear_candidate(self):
        configs = {self.low.key: self.low, self.high.key: self.high}
        self.assertEqual(recorded_decision(configs, Pair("model", self.low.key, self.high.key)), 1)

    def test_holds_clear_incumbent(self):
        configs = {self.low.key: self.low, self.high.key: self.high}
        self.assertEqual(recorded_decision(configs, Pair("model", self.high.key, self.low.key)), -1)

    def test_equal_arms_are_inconclusive(self):
        same = make_config("high2", rewards=np.ones((2, TRIALS), dtype=np.int64))
        configs = {self.high.key: self.high, same.key: same}
        self.assertEqual(recorded_decision(configs, Pair("model", self.high.key, same.key)), 0)


class FullCostTests(unittest.TestCase):
    def test_sums_both_arms(self):
        incumbent = make_config("a", costs=np.full((2, TRIALS), 1.0))
        candidate = make_config("b", costs=np.full((2, TRIALS), 2.0))
        configs = {incumbent.key: incumbent, candidate.key: candidate}
        self.assertEqual(full_cost(configs, Pair("model", incumbent.key, candidate.key)), 24.0)


class RecommendTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(ERROR_CAP, 0.05)
        self.assertEqual(POWER_SLACK, 0.05)

    def test_filters_by_both_caps(self):
        metrics = {
            "aa_promote": [0.04, 0.06, 0.03],
            "worse_promote": [0.04, 0.01, 0.06],
            "better_promote": [0.5, 0.5, 0.5],
            "mean_cost_usd": [1.0, 0.1, 0.1],
        }
        self.assertEqual(recommend(metrics), 0)

    def test_power_slack_and_cost(self):
        metrics = {
            "aa_promote": [0.0, 0.0, 0.0, 0.0],
            "worse_promote": [0.0, 0.0, 0.0, 0.0],
            "better_promote": [0.50, 0.56, 0.44, 0.56],
            "mean_cost_usd": [1.0, 5.0, 0.5, 2.0],
        }
        self.assertEqual(recommend(metrics), 3)

    def test_ties_use_lower_position(self):
        metrics = {
            "aa_promote": [0.0, 0.0],
            "worse_promote": [0.0, 0.0],
            "better_promote": [0.5, 0.5],
            "mean_cost_usd": [1.0, 1.0],
        }
        self.assertEqual(recommend(metrics), 0)

    def test_returns_minus_one_when_none_kept(self):
        metrics = {
            "aa_promote": [0.06, 0.07],
            "worse_promote": [0.0, 0.0],
            "better_promote": [0.5, 0.5],
            "mean_cost_usd": [1.0, 2.0],
        }
        self.assertEqual(recommend(metrics), -1)


class MainTests(unittest.TestCase):
    def test_writes_stable_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "one" / "metrics.json"
            second = Path(folder) / "two" / "metrics.json"
            main(DATA_DIR, first, replicates=5)
            main(DATA_DIR, second, replicates=5)
            data = first.read_bytes()
            self.assertEqual(data, second.read_bytes())
            self.assertTrue(data.endswith(b"\n"))
            metrics = json.loads(data)
            self.assertEqual(len(metrics), 33)
            self.assertEqual(metrics["configs"], 26)
            self.assertEqual(metrics["runs_recorded"], 10832)
            self.assertEqual(metrics["pairs"], 52)
            self.assertEqual(metrics["aa_pairs"], 26)
            self.assertEqual(metrics["replicates"], 5)
            self.assertEqual(metrics["bootstrap_draws"], 1000)
            self.assertEqual(metrics["max_rounds"], 4)
            self.assertEqual(metrics["worse_pairs"] + metrics["better_pairs"], 52)
            recorded_total = (
                metrics["recorded_promote"] + metrics["recorded_hold"] + metrics["recorded_inconclusive"]
            )
            self.assertEqual(recorded_total, 52)
            self.assertEqual(len(metrics["pair_delta"]), 52)
            self.assertEqual(len(metrics["pair_tasks"]), 52)
            self.assertEqual(len(metrics["pair_full_cost_usd"]), 52)
            self.assertEqual(len(metrics["pair_recorded"]), 52)
            self.assertEqual(len(metrics["pair_promote"]), 11 * 52)
            self.assertEqual(len(metrics["pair_hold"]), 11 * 52)
            self.assertEqual(len(metrics["aa_pair_promote"]), 11 * 26)

            per_rule = (
                "aa_promote",
                "aa_hold",
                "aa_inconclusive",
                "worse_promote",
                "worse_hold",
                "worse_inconclusive",
                "better_promote",
                "better_hold",
                "better_inconclusive",
                "mean_runs",
                "mean_cost_usd",
                "cost_share",
            )
            for key in per_rule:
                self.assertEqual(len(metrics[key]), len(RULES), msg=key)

            rate_keys = (
                "aa_promote",
                "aa_hold",
                "aa_inconclusive",
                "worse_promote",
                "worse_hold",
                "worse_inconclusive",
                "better_promote",
                "better_hold",
                "better_inconclusive",
            )
            for key in rate_keys:
                for value in metrics[key]:
                    self.assertGreaterEqual(value, 0.0, msg=key)
                    self.assertLessEqual(value, 1.0, msg=key)
            for value in metrics["pair_promote"] + metrics["pair_hold"] + metrics["aa_pair_promote"]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

            self.assertGreater(metrics["full_cost_usd"], 0.0)
            for i, rule in enumerate(RULES):
                self.assertAlmostEqual(
                    metrics["cost_share"][i],
                    metrics["mean_cost_usd"][i] / metrics["full_cost_usd"],
                    places=5,
                    msg=rule,
                )
            for value in metrics["mean_runs"]:
                self.assertGreater(value, 0.0)
            for value in metrics["mean_cost_usd"]:
                self.assertGreater(value, 0.0)
            self.assertGreaterEqual(metrics["recommended_rule"], -1)
            self.assertLess(metrics["recommended_rule"], len(RULES))

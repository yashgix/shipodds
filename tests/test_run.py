"""Tests for the experiment runner and its recommendation rule."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from promote_or_hold import DATA_DIR
from promote_or_hold.data import TRIALS, Config
from promote_or_hold.gates import ALL_RULES, RULES
from promote_or_hold.pairs import Pair
from promote_or_hold.robustness import Replay
from promote_or_hold.run import V2_NAME, full_cost, main, recommend, recorded_decision
from promote_or_hold.summary import ERROR_CAP, POWER_SLACK, distinct_comparisons


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
            recorded_total = metrics["recorded_promote"] + metrics["recorded_hold"] + metrics["recorded_inconclusive"]
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


class ControlledMainTests(unittest.TestCase):
    def test_main_aggregates_pair_results_and_rule_major_outputs(self):
        first = Config(
            "m1",
            "d",
            "default",
            ("t1", "t2"),
            np.zeros((2, TRIALS), dtype=np.int64),
            np.ones((2, TRIALS)),
        )
        second = Config(
            "m2",
            "d",
            "default",
            ("t1", "t2", "t3"),
            np.zeros((3, TRIALS), dtype=np.int64),
            np.ones((3, TRIALS)),
        )
        configs = {first.key: first, second.key: second}
        real = [
            Pair("model", first.key, second.key),
            Pair("model", second.key, first.key),
            Pair("policy", first.key, first.key),
        ]
        copies = [Pair("aa", first.key, first.key), Pair("aa", second.key, second.key)]

        def summary(index, position):
            promote = 0.001 * (position + 1) + 0.00001 * index
            hold = 0.2
            return {
                "promote": promote,
                "hold": hold,
                "inconclusive": 1.0 - promote - hold,
                "runs": float(10 * (position + 1) + index),
                "cost_usd": float((position + 1) + 0.1 * index),
            }

        def fake_simulate(configs, pair, index, replicates, v2, smoothed):
            self.assertTrue(v2)
            shift = 100 if smoothed else 0
            return {rule: summary(index + shift, position) for position, rule in enumerate(ALL_RULES)}

        def fake_cost(configs, pair):
            del configs, pair
            return 1.0

        def fake_delta(configs, pair):
            del configs, pair
            return 1.0

        def fake_recorded(configs, pair):
            del configs, pair
            return 1

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "nested" / "metrics.json"
            with (
                mock.patch("promote_or_hold.run.load", return_value=configs),
                mock.patch("promote_or_hold.run.real_pairs", return_value=real),
                mock.patch("promote_or_hold.run.aa_pairs", return_value=copies),
                mock.patch("promote_or_hold.run.simulate", side_effect=fake_simulate) as simulate_mock,
                mock.patch("promote_or_hold.run.delta", side_effect=[0.0, -0.1, 0.2]),
                mock.patch("promote_or_hold.run.full_cost", side_effect=[10.0, 20.0, 30.0, 4.0, 6.0]),
                mock.patch("promote_or_hold.run.recorded_decision", side_effect=[1, -1, 0]),
                mock.patch("promote_or_hold.run.v2_metrics", return_value={"v2": 1.0}) as v2_mock,
            ):
                main("ignored", out, replicates=3)
            metrics = json.loads(out.read_text())
            v2 = json.loads(out.with_name(V2_NAME).read_text())

        self.assertEqual(v2, {"v2": 1.0})
        replays, passed_real, passed_copies = v2_mock.call_args.args
        self.assertEqual((passed_real, passed_copies), (real, copies))
        self.assertEqual(set(replays), {"recorded", "smoothed"})
        recorded = replays["recorded"]
        self.assertIsInstance(recorded, Replay)
        self.assertEqual(
            (recorded.deltas, recorded.real_costs, recorded.aa_costs),
            ([0.0, -0.1, 0.2], [10.0, 20.0, 30.0], [4.0, 6.0]),
        )
        self.assertEqual(recorded.aa[0]["mean_1"], summary(3, 0))
        self.assertEqual(replays["smoothed"].real[0]["mean_1"], summary(100, 0))

        self.assertEqual([call.args[2] for call in simulate_mock.call_args_list], [0, 1, 2, 3, 4] * 2)
        self.assertEqual(metrics["configs"], 2)
        self.assertEqual(metrics["runs_recorded"], 20)
        self.assertEqual(metrics["pairs"], 3)
        self.assertEqual(metrics["aa_pairs"], 2)
        self.assertEqual(metrics["worse_pairs"], 2)
        self.assertEqual(metrics["better_pairs"], 1)
        self.assertEqual(metrics["recorded_promote"], 1)
        self.assertEqual(metrics["recorded_hold"], 1)
        self.assertEqual(metrics["recorded_inconclusive"], 1)
        self.assertEqual(metrics["pair_delta"], [0.0, -0.1, 0.2])
        self.assertEqual(metrics["pair_tasks"], [2, 3, 2])
        self.assertEqual(metrics["pair_full_cost_usd"], [10.0, 20.0, 30.0])
        self.assertEqual(metrics["pair_recorded"], [1, -1, 0])

        expected_pair_promote = [
            summary(index, position)["promote"] for position in range(len(RULES)) for index in range(3)
        ]
        expected_pair_hold = [summary(index, position)["hold"] for position in range(len(RULES)) for index in range(3)]
        expected_aa_promote = [
            (summary(3, position)["promote"] + summary(4, position)["promote"]) / 2.0 for position in range(len(RULES))
        ]
        expected_aa_pair_promote = [
            summary(index, position)["promote"] for position in range(len(RULES)) for index in (3, 4)
        ]
        expected_worse_promote = [
            (summary(0, position)["promote"] + summary(1, position)["promote"]) / 2.0 for position in range(len(RULES))
        ]
        expected_better_promote = [summary(2, position)["promote"] for position in range(len(RULES))]
        expected_mean_runs = [
            (summary(0, position)["runs"] + summary(1, position)["runs"] + summary(2, position)["runs"]) / 3.0
            for position in range(len(RULES))
        ]
        expected_mean_cost = [
            (summary(0, position)["cost_usd"] + summary(1, position)["cost_usd"] + summary(2, position)["cost_usd"])
            / 3.0
            for position in range(len(RULES))
        ]
        expected_cost_share = [cost / 20.0 for cost in expected_mean_cost]

        np.testing.assert_allclose(metrics["pair_promote"], expected_pair_promote, atol=1e-8)
        np.testing.assert_allclose(metrics["pair_hold"], expected_pair_hold, atol=1e-8)
        np.testing.assert_allclose(metrics["aa_pair_promote"], expected_aa_pair_promote, atol=1e-8)
        np.testing.assert_allclose(metrics["aa_promote"], expected_aa_promote, atol=1e-8)
        np.testing.assert_allclose(metrics["worse_promote"], expected_worse_promote, atol=1e-8)
        np.testing.assert_allclose(metrics["better_promote"], expected_better_promote, atol=1e-8)
        np.testing.assert_allclose(metrics["mean_runs"], expected_mean_runs, atol=1e-8)
        np.testing.assert_allclose(metrics["mean_cost_usd"], expected_mean_cost, atol=1e-8)
        np.testing.assert_allclose(metrics["cost_share"], expected_cost_share, atol=1e-8)
        self.assertEqual(metrics["recommended_rule"], 0)


class DistinctComparisonTests(unittest.TestCase):
    def test_reverse_pair_is_the_same_comparison(self):
        a, b, c = ("a", "d", "default"), ("b", "d", "default"), ("c", "d", "default")
        pairs = [Pair("model", a, b), Pair("model", b, a), Pair("model", a, c)]
        self.assertEqual(distinct_comparisons(pairs), 2)

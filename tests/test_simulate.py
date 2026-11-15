"""Tests for the replay simulation."""

import unittest

import numpy as np

from promote_or_hold import DATA_DIR
from promote_or_hold.data import TRIALS, Config, load
from promote_or_hold.gates import OBRIEN_FLEMING, POCOCK, ROUNDS, RULES, sequential_gate
from promote_or_hold.pairs import Pair, aa_pairs
from promote_or_hold.simulate import FIXED_TRIALS, draw, outcomes, simulate, weights_for


def make_config(model="m", domain="d", mode="default", tasks=("t1", "t2"), rewards=None, costs=None):
    tasks = tuple(tasks)
    if rewards is None:
        rewards = np.zeros((len(tasks), TRIALS), dtype=np.int64)
    if costs is None:
        costs = np.ones((len(tasks), TRIALS), dtype=float)
    return Config(model, domain, mode, tasks, np.asarray(rewards, dtype=np.int64), np.asarray(costs, dtype=float))


class DrawTests(unittest.TestCase):
    def setUp(self):
        rewards_a = np.array([[1, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int64)
        costs_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        rewards_b = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.int64)
        costs_b = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=float)
        self.a = make_config("a", "d", "default", rewards=rewards_a, costs=costs_a)
        self.b = make_config("b", "d", "default", rewards=rewards_b, costs=costs_b)

    def test_shapes_and_dtypes(self):
        ya, yb, cost = draw(self.a, self.b, np.random.default_rng(0), 6)
        for array in (ya, yb, cost):
            self.assertEqual(array.shape, (6, 2, ROUNDS))
        self.assertEqual(ya.dtype.kind, "i")
        self.assertEqual(yb.dtype.kind, "i")
        self.assertEqual(cost.dtype.kind, "f")

    def test_values_come_from_recorded_trials(self):
        ya, yb, cost = draw(self.a, self.b, np.random.default_rng(1), 30)
        for t in range(2):
            self.assertTrue(set(np.unique(ya[:, t, :]).tolist()) <= set(self.a.rewards[t].tolist()))
            self.assertTrue(set(np.unique(yb[:, t, :]).tolist()) <= set(self.b.rewards[t].tolist()))
            expected = {self.a.costs[t, i] + self.b.costs[t, j] for i in range(TRIALS) for j in range(TRIALS)}
            for value in np.unique(cost[:, t, :]):
                self.assertIn(value, expected)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        zeros = np.zeros((2, TRIALS), dtype=np.int64)
        ones = np.ones((2, TRIALS), dtype=np.int64)
        self.a = make_config("a", "d", "default", rewards=zeros, costs=np.ones((2, TRIALS), dtype=float))
        self.b = make_config("b", "d", "default", rewards=ones, costs=np.full((2, TRIALS), 2.0))
        self.ya, self.yb, self.cost = draw(self.a, self.b, np.random.default_rng(7), 5)
        self.results = outcomes(self.ya, self.yb, self.cost, weights_for(2))

    def test_all_rules_and_shapes(self):
        self.assertEqual(list(self.results), list(RULES))
        for decisions, runs, usd in self.results.values():
            self.assertEqual(decisions.shape, (5,))
            self.assertEqual(runs.shape, (5,))
            self.assertEqual(usd.shape, (5,))
            self.assertEqual(decisions.dtype.kind, "i")
            self.assertEqual(runs.dtype.kind, "i")
            self.assertEqual(usd.dtype.kind, "f")

    def test_fixed_runs_and_costs(self):
        for k in FIXED_TRIALS:
            for prefix in ("mean", "bootstrap", "cmh"):
                _, runs, usd = self.results[f"{prefix}_{k}"]
                np.testing.assert_array_equal(runs, np.full(5, 2 * 2 * k))
                np.testing.assert_allclose(usd, self.cost[:, :, :k].sum(axis=(1, 2)))

    def test_sequential_runs_and_costs(self):
        for name, bounds in (("sequential_pocock", POCOCK), ("sequential_obf", OBRIEN_FLEMING)):
            decisions, runs, usd = self.results[name]
            expected_decisions, rounds = sequential_gate(self.ya, self.yb, bounds)
            np.testing.assert_array_equal(decisions, expected_decisions)
            np.testing.assert_array_equal(runs, 2 * 2 * rounds)
            mask = np.arange(ROUNDS)[None, :] < rounds[:, None]
            np.testing.assert_allclose(usd, (self.cost * mask[:, None, :]).sum(axis=(1, 2)))

    def test_deterministic_rewards(self):
        for k in FIXED_TRIALS:
            np.testing.assert_array_equal(self.results[f"mean_{k}"][0], np.ones(5, dtype=np.int64))
            np.testing.assert_array_equal(self.results[f"bootstrap_{k}"][0], np.ones(5, dtype=np.int64))
        np.testing.assert_array_equal(self.results["cmh_1"][0], np.zeros(5, dtype=np.int64))
        np.testing.assert_array_equal(self.results["cmh_2"][0], np.ones(5, dtype=np.int64))
        np.testing.assert_array_equal(self.results["cmh_4"][0], np.ones(5, dtype=np.int64))
        np.testing.assert_array_equal(self.results["sequential_pocock"][0], np.ones(5, dtype=np.int64))
        np.testing.assert_array_equal(self.results["sequential_obf"][0], np.ones(5, dtype=np.int64))


class SimulateTests(unittest.TestCase):
    def setUp(self):
        rewards_a = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int64)
        costs_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        rewards_b = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
        costs_b = np.array([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=float)
        self.a = make_config("a", "d", "default", rewards=rewards_a, costs=costs_a)
        self.b = make_config("b", "d", "default", rewards=rewards_b, costs=costs_b)
        self.configs = {self.a.key: self.a, self.b.key: self.b}
        self.pair = Pair("model", self.a.key, self.b.key)

    def test_seed_and_index_reproducible(self):
        first = simulate(self.configs, self.pair, index=3, replicates=20)
        second = simulate(self.configs, self.pair, index=3, replicates=20)
        self.assertEqual(first, second)

    def test_result_shape_and_share_sum(self):
        result = simulate(self.configs, self.pair, index=4, replicates=20)
        self.assertEqual(set(result), set(RULES))
        for rule in RULES:
            metrics = result[rule]
            self.assertEqual(set(metrics), {"promote", "hold", "inconclusive", "runs", "cost_usd"})
            total = metrics["promote"] + metrics["hold"] + metrics["inconclusive"]
            self.assertAlmostEqual(total, 1.0)
            self.assertGreaterEqual(metrics["promote"], 0.0)
            self.assertGreaterEqual(metrics["hold"], 0.0)
            self.assertGreaterEqual(metrics["inconclusive"], 0.0)
            self.assertGreater(metrics["runs"], 0.0)
            self.assertGreater(metrics["cost_usd"], 0.0)
        self.assertIs(weights_for(3), weights_for(3))

    def test_mismatched_tasks_raise(self):
        other = make_config(
            "c",
            "d",
            "default",
            tasks=("t1",),
            rewards=np.zeros((1, TRIALS), dtype=np.int64),
            costs=np.ones((1, TRIALS), dtype=float),
        )
        configs = {self.a.key: self.a, other.key: other}
        with self.assertRaises(ValueError):
            simulate(configs, Pair("model", self.a.key, other.key), index=0, replicates=2)


class RealAATest(unittest.TestCase):
    def test_small_replicate_run_on_real_aa_pair(self):
        configs = load(DATA_DIR)
        pair = aa_pairs(configs)[0]
        result = simulate(configs, pair, index=52, replicates=8)
        for rule in RULES:
            metrics = result[rule]
            total = metrics["promote"] + metrics["hold"] + metrics["inconclusive"]
            self.assertAlmostEqual(total, 1.0)
            self.assertGreater(metrics["runs"], 0.0)
            self.assertGreater(metrics["cost_usd"], 0.0)

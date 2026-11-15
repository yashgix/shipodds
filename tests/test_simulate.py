"""Tests for the replay simulation."""

import unittest

import numpy as np

from promote_or_hold import DATA_DIR
from promote_or_hold.data import TRIALS, Config, load
from promote_or_hold.gates import (
    FUTILITY,
    NAIVE_PEEK,
    OBRIEN_FLEMING,
    POCOCK,
    ROUNDS,
    RULES,
    V2_RULES,
    bootstrap_weights,
    sequential_gate,
)
from promote_or_hold.pairs import Pair, aa_pairs
from promote_or_hold.simulate import (
    FIXED_TRIALS,
    SEED,
    draw,
    draw_smoothed,
    outcomes,
    simulate,
    v2_outcomes,
    weights_for,
)


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

    def test_exact_seeded_draw_uses_incumbent_then_candidate(self):
        rng = np.random.default_rng(11)
        expected_ia = rng.integers(0, TRIALS, (6, 2, ROUNDS))
        expected_ib = rng.integers(0, TRIALS, (6, 2, ROUNDS))
        tasks = np.arange(2)[None, :, None]
        expected_ya = self.a.rewards[tasks, expected_ia]
        expected_yb = self.b.rewards[tasks, expected_ib]
        expected_cost = self.a.costs[tasks, expected_ia] + self.b.costs[tasks, expected_ib]
        ya, yb, cost = draw(self.a, self.b, np.random.default_rng(11), 6)
        np.testing.assert_array_equal(ya, expected_ya)
        np.testing.assert_array_equal(yb, expected_yb)
        np.testing.assert_allclose(cost, expected_cost)

    def test_identical_copy_arms_draw_independently(self):
        rng = np.random.default_rng(12)
        expected_ia = rng.integers(0, TRIALS, (8, 2, ROUNDS))
        expected_ib = rng.integers(0, TRIALS, (8, 2, ROUNDS))
        tasks = np.arange(2)[None, :, None]
        expected_ya = self.a.rewards[tasks, expected_ia]
        expected_yb = self.a.rewards[tasks, expected_ib]
        ya, yb, _ = draw(self.a, self.a, np.random.default_rng(12), 8)
        np.testing.assert_array_equal(ya, expected_ya)
        np.testing.assert_array_equal(yb, expected_yb)
        self.assertFalse(np.array_equal(expected_ia, expected_ib))


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

    def test_weights_for_matches_seeded_bootstrap_weights(self):
        expected = bootstrap_weights(3, np.random.default_rng([SEED, 0, 3]))
        np.testing.assert_array_equal(weights_for(3), expected)
        self.assertIs(weights_for(3), weights_for(3))

    def test_simulate_uses_seeded_draw_and_index(self):
        index = 5
        replicates = 7
        rng = np.random.default_rng([SEED, 1, index])
        ya, yb, cost = draw(self.a, self.b, rng, replicates)
        results = outcomes(ya, yb, cost, weights_for(len(self.a.tasks)))
        result = simulate(self.configs, self.pair, index=index, replicates=replicates)
        for rule in RULES:
            decisions, runs, usd = results[rule]
            expected = {
                "promote": float(np.mean(decisions == 1)),
                "hold": float(np.mean(decisions == -1)),
                "inconclusive": float(np.mean(decisions == 0)),
                "runs": float(np.mean(runs)),
                "cost_usd": float(np.mean(usd)),
            }
            for key, value in expected.items():
                self.assertAlmostEqual(result[rule][key], value, places=12)

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


class V2RuleTests(unittest.TestCase):
    def setUp(self):
        rewards = np.array([[1, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int64)
        self.a = make_config("a", rewards=rewards)
        self.b = make_config("b", rewards=rewards[::-1])
        self.ya, self.yb, self.cost = draw(self.a, self.b, np.random.default_rng(3), 6)

    def test_always_hold_never_promotes_and_costs_nothing(self):
        decisions, runs, usd = v2_outcomes(self.ya, self.yb, self.cost)["always_hold"]
        np.testing.assert_array_equal(decisions, np.full(6, -1))
        np.testing.assert_array_equal(runs, np.zeros(6))
        np.testing.assert_array_equal(usd, np.zeros(6))

    def test_naive_peek_is_the_sequential_gate_at_the_fixed_threshold(self):
        decisions, runs, usd = v2_outcomes(self.ya, self.yb, self.cost)["naive_peek"]
        expected, rounds = sequential_gate(self.ya, self.yb, NAIVE_PEEK)
        np.testing.assert_array_equal(decisions, expected)
        np.testing.assert_array_equal(runs, 2 * 2 * rounds)
        mask = np.arange(ROUNDS)[None, :] < rounds[:, None]
        np.testing.assert_allclose(usd, (self.cost * mask[:, None, :]).sum(axis=(1, 2)))

    def test_futility_rules_are_obf_with_their_threshold(self):
        results = v2_outcomes(self.ya, self.yb, self.cost)
        for rule, threshold in FUTILITY.items():
            expected, rounds = sequential_gate(self.ya, self.yb, OBRIEN_FLEMING, threshold)
            np.testing.assert_array_equal(results[rule][0], expected)
            np.testing.assert_array_equal(results[rule][1], 2 * 2 * rounds)

    def test_simulate_adds_v2_rules_only_when_asked(self):
        configs = {self.a.key: self.a, self.b.key: self.b}
        pair = Pair("model", self.a.key, self.b.key)
        v1 = simulate(configs, pair, 0, replicates=4)
        both = simulate(configs, pair, 0, replicates=4, v2=True)
        self.assertEqual(list(v1), list(RULES))
        self.assertEqual(list(both), list(RULES + V2_RULES))
        self.assertEqual({rule: both[rule] for rule in RULES}, v1)

    def test_peeking_inflates_false_promotions_on_an_identical_copy(self):
        # Every task passes half its trials, and both arms are the same agent.
        config = make_config(tasks=[f"t{i}" for i in range(100)], rewards=np.tile([1, 0, 1, 0], (100, 1)))
        ya, yb, cost = draw(config, config, np.random.default_rng(11), 4000)
        peek = np.mean(v2_outcomes(ya, yb, cost)["naive_peek"][0] == 1)
        obf = np.mean(outcomes(ya, yb, cost, weights_for(100))["sequential_obf"][0] == 1)
        self.assertLess(abs(obf - 0.025), 0.01)
        self.assertGreater(peek, 0.045)


class SmoothedTests(unittest.TestCase):
    def setUp(self):
        costs = np.array([[1.0, 3.0, 1.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
        self.a = make_config("a", rewards=np.zeros((2, TRIALS)), costs=costs)
        self.b = make_config("b", rewards=np.array([[1, 1, 1, 1], [1, 0, 1, 0]]), costs=costs)

    def test_rates_and_costs(self):
        ya, yb, cost = draw_smoothed(self.a, self.b, np.random.default_rng(5), 20000)
        self.assertEqual(ya.shape, (20000, 2, ROUNDS))
        np.testing.assert_allclose(ya.mean(axis=(0, 2)), [0.1, 0.1], atol=0.01)
        np.testing.assert_allclose(yb.mean(axis=(0, 2)), [0.9, 0.5], atol=0.01)
        np.testing.assert_array_equal(cost[0, :, 0], [4.0, 4.0])

    def test_simulate_uses_a_separate_stream(self):
        configs = {self.a.key: self.a, self.b.key: self.b}
        pair = Pair("model", self.a.key, self.b.key)
        recorded = simulate(configs, pair, 0, replicates=50)
        smoothed = simulate(configs, pair, 0, replicates=50, smoothed=True)
        self.assertEqual(smoothed, simulate(configs, pair, 0, replicates=50, smoothed=True))
        self.assertNotEqual(recorded["mean_1"], smoothed["mean_1"])


class UnsignedRewardTests(unittest.TestCase):
    def test_unsigned_rewards_simulate_like_signed(self):
        rewards_a = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1]])
        rewards_b = np.zeros((3, TRIALS))

        def configs(dtype):
            a = Config("a", "d", "default", ("t1", "t2", "t3"), rewards_a.astype(dtype), np.ones((3, TRIALS)))
            b = Config("b", "d", "default", ("t1", "t2", "t3"), rewards_b.astype(dtype), np.ones((3, TRIALS)))
            return {a.key: a, b.key: b}, Pair("model", a.key, b.key)

        signed = simulate(*configs(np.int64), 0, replicates=200, v2=True)
        unsigned = simulate(*configs(np.uint8), 0, replicates=200, v2=True)
        self.assertEqual(unsigned, signed)
        self.assertEqual(unsigned["bootstrap_4"]["promote"], 0.0)

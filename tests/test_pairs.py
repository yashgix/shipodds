"""Tests for the real comparison pairs and the true difference."""

import itertools
import unittest

import numpy as np

from promote_or_hold import DATA_DIR
from promote_or_hold.data import TRIALS, Config, load
from promote_or_hold.pairs import DOMAINS, MODES, POLICY_MODELS, Pair, aa_pairs, delta, real_pairs


class RealPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load(DATA_DIR)
        cls.pairs = real_pairs(cls.configs)
        cls.copies = aa_pairs(cls.configs)

    def test_counts(self):
        self.assertEqual(len(self.pairs), 52)
        self.assertEqual(sum(pair.kind == "model" for pair in self.pairs), 36)
        self.assertEqual(sum(pair.kind == "policy" for pair in self.pairs), 16)
        self.assertEqual(len(self.copies), 26)

    def test_real_pair_order_and_kinds(self):
        models = sorted({model for model, _, _ in self.configs})
        expected = []
        for domain in DOMAINS:
            for a, b in itertools.permutations(models, 2):
                expected.append(Pair("model", (a, domain, "default"), (b, domain, "default")))
        for model in POLICY_MODELS:
            for mode in MODES:
                expected.append(Pair("policy", (model, "telecom", mode), (model, "telecom-workflow", mode)))
                expected.append(Pair("policy", (model, "telecom-workflow", mode), (model, "telecom", mode)))
        self.assertEqual(self.pairs, expected)
        self.assertEqual([pair.kind for pair in self.pairs[:36]], ["model"] * 36)
        self.assertEqual([pair.kind for pair in self.pairs[36:]], ["policy"] * 16)

    def test_aa_pairs_are_sorted_identical_copies(self):
        self.assertEqual(self.copies, [Pair("aa", key, key) for key in sorted(self.configs)])
        for pair in self.copies:
            self.assertEqual(pair.incumbent, pair.candidate)
            self.assertEqual(delta(self.configs, pair), 0.0)

    def test_keys_exist_and_tasks_match(self):
        for pair in self.pairs + self.copies:
            incumbent = self.configs[pair.incumbent]
            candidate = self.configs[pair.candidate]
            self.assertEqual(incumbent.tasks, candidate.tasks, msg=pair)

    def test_reverse_deltas_cancel(self):
        for pair in self.pairs:
            reverse = Pair(pair.kind, pair.candidate, pair.incumbent)
            self.assertAlmostEqual(delta(self.configs, pair) + delta(self.configs, reverse), 0.0, places=12)


class DeltaTests(unittest.TestCase):
    def test_delta_is_candidate_mean_minus_incumbent_mean(self):
        incumbent = Config(
            "m",
            "d",
            "default",
            ("t1", "t2"),
            np.array([[1, 0], [0, 0]], dtype=np.int64),
            np.ones((2, TRIALS)),
        )
        candidate = Config(
            "m",
            "d",
            "other",
            ("t1", "t2"),
            np.array([[1, 1], [0, 0]], dtype=np.int64),
            np.ones((2, TRIALS)),
        )
        configs = {config.key: config for config in (incumbent, candidate)}
        self.assertAlmostEqual(delta(configs, Pair("model", incumbent.key, candidate.key)), 0.25)
        self.assertAlmostEqual(delta(configs, Pair("model", candidate.key, incumbent.key)), -0.25)

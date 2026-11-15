import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import promote_or_hold
from promote_or_hold.data import TRIALS, Config
from promote_or_hold.gates import ROUNDS, RULES
from promote_or_hold.pairs import Pair
from promote_or_hold.run import main, recommend
from promote_or_hold.simulate import REPLICATES, SEED, draw, simulate

ROOT = Path(promote_or_hold.__file__).resolve().parent.parent
GROUPS = ("aa", "worse", "better")
PER_RULE = [f"{g}_{o}" for g in GROUPS for o in ("promote", "hold", "inconclusive")] + [
    "mean_runs",
    "mean_cost_usd",
    "cost_share",
]
SCALARS = (
    "configs",
    "runs_recorded",
    "pairs",
    "aa_pairs",
    "worse_pairs",
    "better_pairs",
    "replicates",
    "bootstrap_draws",
    "max_rounds",
    "recommended_rule",
    "full_cost_usd",
    "recorded_promote",
    "recorded_hold",
    "recorded_inconclusive",
)
PER_PAIR = {
    "pair_delta": 52,
    "pair_tasks": 52,
    "pair_full_cost_usd": 52,
    "pair_recorded": 52,
    "pair_promote": 572,
    "pair_hold": 572,
    "aa_pair_promote": 286,
}


def config(name, rewards, costs, tasks=None):
    rewards = np.asarray(rewards)
    tasks = tasks or tuple(f"t{i}" for i in range(len(rewards)))
    return Config(name, "d", "default", tasks, rewards, np.broadcast_to(costs, rewards.shape).astype(float))


def two(a, b):
    return {a.key: a, b.key: b}, Pair("model", a.key, b.key)


class Simulation(unittest.TestCase):
    def setUp(self):
        self.low = config("low", np.zeros((5, TRIALS), int), 1.0)
        self.high = config("high", np.ones((5, TRIALS), int), 2.0)

    def test_certain_improvement(self):
        configs, pair = two(self.low, self.high)
        out = simulate(configs, pair, 0, replicates=50)
        self.assertEqual(set(out), set(RULES))
        for rule, r in out.items():
            self.assertEqual((r["promote"], r["hold"], r["inconclusive"]), (1.0, 0.0, 0.0), rule)
        for k in (1, 2, 4):
            for rule in (f"mean_{k}", f"bootstrap_{k}", f"cmh_{k}"):
                self.assertEqual((out[rule]["runs"], out[rule]["cost_usd"]), (10.0 * k, 15.0 * k), rule)
        for rule in ("sequential_pocock", "sequential_obf"):  # both cross at round 2 (z = sqrt(5), then sqrt(15))
            self.assertEqual((out[rule]["runs"], out[rule]["cost_usd"]), (20.0, 30.0), rule)

    def test_certain_regression(self):
        configs, pair = two(self.high, self.low)
        for rule, r in simulate(configs, pair, 0, replicates=50).items():
            self.assertEqual((r["promote"], r["hold"], r["inconclusive"]), (0.0, 1.0, 0.0), rule)

    def test_identical_constant_configs(self):
        configs = {self.low.key: self.low}
        out = simulate(configs, Pair("aa", self.low.key, self.low.key), 0, replicates=50)
        for rule, r in out.items():
            expected = (0.0, 1.0, 0.0) if rule.startswith("mean") else (0.0, 0.0, 1.0)
            self.assertEqual((r["promote"], r["hold"], r["inconclusive"]), expected, rule)
        self.assertEqual((out["sequential_obf"]["runs"], out["sequential_obf"]["cost_usd"]), (40.0, 40.0))

    def test_draw_protocol(self):
        rng = np.random.default_rng(0)
        a = config("a", rng.integers(0, 2, (6, TRIALS)), rng.random((6, TRIALS)))
        b = config("b", rng.integers(0, 2, (6, TRIALS)), rng.random((6, TRIALS)))
        ya, yb, cost = draw(a, b, np.random.default_rng(9), 7)
        check = np.random.default_rng(9)
        ia = check.integers(0, TRIALS, (7, 6, ROUNDS))
        ib = check.integers(0, TRIALS, (7, 6, ROUNDS))
        t = np.arange(6)[None, :, None]
        np.testing.assert_array_equal(ya, a.rewards[t, ia])
        np.testing.assert_array_equal(yb, b.rewards[t, ib])
        np.testing.assert_allclose(cost, a.costs[t, ia] + b.costs[t, ib], atol=0)

    def test_seeded_by_pair_index(self):
        rng = np.random.default_rng(1)
        a = config("a", rng.integers(0, 2, (30, TRIALS)), 1.0)
        b = config("b", rng.integers(0, 2, (30, TRIALS)), 1.0)
        configs, pair = two(a, b)
        self.assertEqual(SEED, 20261114)
        self.assertEqual(simulate(configs, pair, 3, 60), simulate(configs, pair, 3, 60))
        self.assertNotEqual(simulate(configs, pair, 3, 60), simulate(configs, pair, 4, 60))

    def test_tasks_must_match(self):
        other = config("other", np.ones((5, TRIALS), int), 1.0, tasks=tuple("abcde"))
        configs, pair = two(self.low, other)
        with self.assertRaises(ValueError):
            simulate(configs, pair, 0, replicates=5)


class Recommend(unittest.TestCase):
    def metrics(self, aa, worse, better, cost):
        return {"aa_promote": aa, "worse_promote": worse, "better_promote": better, "mean_cost_usd": cost}

    def test_cheapest_rule_near_the_best_within_the_caps(self):
        m = self.metrics(
            aa=[0.4, 0.02, 0.02, 0.05, 0.02],
            worse=[0.07, 0.0, 0.06, 0.0, 0.0],
            better=[0.9, 0.7, 0.8, 0.66, 0.64],
            cost=[10, 90, 20, 50, 40],
        )
        self.assertEqual(recommend(m), 3)
        m["mean_cost_usd"][1] = 50
        self.assertEqual(recommend(m), 1)

    def test_none_within_the_caps(self):
        self.assertEqual(recommend(self.metrics([0.4, 0.06], [0.0, 0.0], [0.9, 0.5], [1, 2])), -1)


class Experiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        main(out=out / "a.json", replicates=100)
        main(out=out / "b.json", replicates=100)
        main(out=out / "results" / "metrics.json")
        cls.text = (out / "a.json").read_text()
        cls.small = json.loads(cls.text)
        cls.full = json.loads((out / "results" / "metrics.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metric_names_and_lengths(self):
        m = self.full
        self.assertEqual(set(m), set(SCALARS) | set(PER_RULE) | set(PER_PAIR))
        for name in SCALARS:
            self.assertIsInstance(m[name], int | float, name)
        for name in PER_RULE:
            self.assertEqual(len(m[name]), len(RULES), name)
        for name, n in PER_PAIR.items():
            self.assertEqual(len(m[name]), n, name)

    def test_identical_sorted_rounded_output(self):
        out = Path(self.tmp.name)
        self.assertEqual(self.text, (out / "b.json").read_text())
        self.assertEqual(self.text, json.dumps(self.small, sort_keys=True, indent=1) + "\n")
        for v in self.small["mean_cost_usd"] + self.small["pair_delta"]:
            self.assertEqual(v, round(v, 6))

    def test_counts(self):
        m = self.full
        self.assertEqual(
            [m[k] for k in SCALARS if k not in ("recommended_rule", "full_cost_usd")],
            [26, 10832, 52, 26, 26, 26, REPLICATES, 1000, 4, 19, 19, 14],
        )
        self.assertEqual(self.small["replicates"], 100)
        self.assertEqual(m["mean_runs"][:9], [198.461538, 396.923077, 793.846154] * 3)
        self.assertAlmostEqual(m["full_cost_usd"], 131.03519, places=5)
        self.assertEqual(m["pair_tasks"], [50] * 12 + [114] * 40)
        self.assertEqual(m["pair_recorded"][:3], [0, 0, 1])

    def test_values_at_100_replicates(self):
        m = self.small
        self.assertEqual(m["aa_promote"][:4], [0.421923, 0.461154, 0.464615, 0.025])
        self.assertEqual(m["worse_promote"][:3], [0.083077, 0.058077, 0.038077])
        self.assertEqual(m["better_promote"][8:], [0.700769, 0.634615, 0.689231])
        self.assertEqual(m["mean_cost_usd"][9:], [80.810237, 94.697751])
        self.assertEqual(m["pair_promote"][:3], [0.8, 0.47, 0.94])
        self.assertEqual(m["aa_pair_promote"][:3], [0.49, 0.44, 0.4])

    def test_full_run_headline(self):
        m = self.full
        i, j = RULES.index("mean_1"), RULES.index("sequential_obf")
        self.assertEqual((m["aa_promote"][i], m["worse_promote"][i]), (0.438923, 0.074731))
        self.assertEqual((m["aa_promote"][j], m["better_promote"][j]), (0.023942, 0.692404))
        self.assertEqual(m["better_promote"][RULES.index("cmh_4")], 0.701615)
        self.assertEqual((m["mean_cost_usd"][j], m["cost_share"][j]), (93.649306, 0.714688))
        self.assertEqual(m["recommended_rule"], j)

    def test_group_outcomes_add_up(self):
        for g in GROUPS:
            for i in range(len(RULES)):
                total = sum(self.full[f"{g}_{o}"][i] for o in ("promote", "hold", "inconclusive"))
                self.assertAlmostEqual(total, 1.0, places=5)

    def test_pair_lists_match_the_groups(self):
        m, n = self.full, 52
        worse = [j for j in range(n) if m["pair_delta"][j] <= 0]
        for i in range(len(RULES)):
            promote = [m["pair_promote"][i * n + j] for j in worse]
            self.assertAlmostEqual(np.mean(promote), m["worse_promote"][i], places=5)
            self.assertAlmostEqual(np.mean(m["aa_pair_promote"][i * 26 : (i + 1) * 26]), m["aa_promote"][i], places=5)
            self.assertAlmostEqual(m["cost_share"][i], m["mean_cost_usd"][i] / m["full_cost_usd"], places=5)

    def test_command_line(self):
        out = Path(self.tmp.name) / "cli.json"
        args = [
            sys.executable,
            "experiment.py",
            "--out",
            str(out),
            "--replicates",
            "20",
            "--data",
            str(promote_or_hold.DATA_DIR),
        ]
        subprocess.run(args, cwd=ROOT, check=True, capture_output=True, timeout=120)
        self.assertEqual(json.loads(out.read_text())["replicates"], 20)

    def test_generated_readme(self):
        folder = Path(self.tmp.name)
        readme = folder / "README.in.md"
        readme.write_text("# Title\n\nBefore.\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\nAfter.\n")
        args = [
            sys.executable,
            "demo.py",
            "--readme",
            str(readme),
            "--out",
            str(folder / "README.out.md"),
            "--results",
            str(folder / "results"),
        ]
        subprocess.run(args, cwd=ROOT, check=True, capture_output=True, timeout=120)
        text = (folder / "README.out.md").read_text()
        self.assertTrue(text.startswith("# Title\n\nBefore.\n\n<!-- results:start -->\n"))
        self.assertTrue(text.endswith("<!-- results:end -->\n\nAfter.\n"))
        self.assertNotIn("\nold\n", text)
        for rule in RULES:
            self.assertIn(rule, text)
        self.assertIn("43.9%", text)
        self.assertIn("**sequential_obf** (recommended)", text)
        summary = subprocess.run(
            [sys.executable, "demo.py", "--results", str(folder / "results")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        self.assertEqual(len(summary.strip().splitlines()), 1)

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import promote_or_hold
from promote_or_hold.data import TRIALS, Config, load
from promote_or_hold.pairs import Pair, aa_pairs, delta, real_pairs
from promote_or_hold.run import recorded_decision

DATA = promote_or_hold.DATA_DIR
MODELS = ("claude-3-7-sonnet-20250219", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "o4-mini-2025-04-16")
SUCCESSES = {
    ("claude-3-7-sonnet-20250219", "airline", "default"): 100,
    ("gpt-4.1-2025-04-14", "telecom", "default"): 156,
    ("gpt-4.1-2025-04-14", "telecom-workflow", "default"): 236,
    ("gpt-4.1-mini-2025-04-14", "retail", "default"): 301,
    ("o4-mini-2025-04-16", "telecom", "no-user-op"): 451,
}
RECORDED = [
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    -1,
    0,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    0,
    1,
    1,
    1,
    1,
    0,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    0,
    1,
    -1,
    0,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    0,
    0,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
]
HEADER = "model,domain,mode,task_id,trial,reward,agent_cost,user_cost\n"


def write_csv(folder, rows):
    path = Path(folder) / "runs.csv"
    path.write_text(HEADER + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return folder


class Files(unittest.TestCase):
    def test_runs_match_the_manifest(self):
        manifest = json.loads((DATA / "MANIFEST.json").read_text())
        data = (DATA / "runs.csv").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["runs.csv"]["sha256"])
        self.assertEqual(
            manifest["runs.csv"]["sha256"], "a7b198ba04bee6ad53bc4522c05033cc35af3717aeb16d4d79bbb695939a54e1"
        )
        self.assertEqual((len(data), data.count(b"\n") - 1), (1712685, 10832))

    def test_manifest_pins_the_upstream_commit(self):
        manifest = json.loads((DATA / "MANIFEST.json").read_text())
        self.assertEqual(manifest["commit"], "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4")
        self.assertEqual(len(manifest["upstream"]), 26)
        self.assertEqual(sum(e["bytes"] for e in manifest["upstream"].values()), 603837972)

    def test_license_copy(self):
        text = (DATA / "tau2-bench-LICENSE").read_text()
        self.assertTrue(text.startswith("MIT License"))
        self.assertIn("Sierra Research", text)


class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load()

    def test_configs(self):
        c = self.configs
        self.assertEqual(len(c), 26)
        self.assertEqual(list(c), sorted(c))
        self.assertEqual(sum(x.rewards.size for x in c.values()), 10832)
        self.assertEqual({k[0] for k in c}, set(MODELS))
        self.assertNotIn("base", {k[2] for k in c})
        self.assertIn(("gpt-4.1-mini-2025-04-14", "airline", "default"), c)

    def test_shapes_and_values(self):
        for key, x in self.configs.items():
            self.assertIsInstance(x, Config)
            self.assertEqual(x.key, key)
            self.assertEqual((x.model, x.domain, x.mode), key)
            self.assertEqual(x.rewards.shape, (50 if key[1] == "airline" else 114, TRIALS))
            self.assertEqual(x.costs.shape, x.rewards.shape)
            self.assertTrue(np.issubdtype(x.rewards.dtype, np.integer))
            self.assertTrue(set(np.unique(x.rewards)) <= {0, 1})
            self.assertTrue(np.all(x.costs > 0))
            self.assertEqual(list(x.tasks), sorted(x.tasks))

    def test_success_counts(self):
        for key, n in SUCCESSES.items():
            self.assertEqual(int(self.configs[key].rewards.sum()), n, key)

    def test_costs_are_agent_plus_user(self):
        x = self.configs[("claude-3-7-sonnet-20250219", "airline", "default")]
        self.assertEqual(x.tasks[:3], ("0", "1", "10"))
        np.testing.assert_allclose(x.costs[0], [0.105075, 0.122155, 0.170697, 0.1741], atol=1e-9)
        self.assertEqual(x.rewards[0].tolist(), [1, 0, 0, 1])
        self.assertAlmostEqual(sum(float(c.costs.sum()) for c in self.configs.values()), 1491.55143, places=4)

    def test_telecom_variants_share_tasks(self):
        tasks = {x.tasks for k, x in self.configs.items() if k[1].startswith("telecom")}
        self.assertEqual(len(tasks), 1)


class Pairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load()
        cls.real = real_pairs(cls.configs)

    def test_real_pairs(self):
        r = self.real
        self.assertEqual(len(r), 52)
        self.assertEqual([p.kind for p in r], ["model"] * 36 + ["policy"] * 16)
        c, g = MODELS[0], MODELS[1]
        self.assertEqual(r[0], Pair("model", (c, "airline", "default"), (g, "airline", "default")))
        self.assertEqual(r[35].incumbent, ("o4-mini-2025-04-16", "telecom", "default"))
        self.assertEqual(r[35].candidate, ("gpt-4.1-mini-2025-04-14", "telecom", "default"))
        self.assertEqual(r[36], Pair("policy", (g, "telecom", "default"), (g, "telecom-workflow", "default")))
        self.assertEqual(r[37], Pair("policy", (g, "telecom-workflow", "default"), (g, "telecom", "default")))
        o = "o4-mini-2025-04-16"
        self.assertEqual(r[51], Pair("policy", (o, "telecom-workflow", "no-user-op"), (o, "telecom", "no-user-op")))

    def test_aa_pairs(self):
        aa = aa_pairs(self.configs)
        self.assertEqual(len(aa), 26)
        self.assertTrue(all(p.kind == "aa" and p.incumbent == p.candidate for p in aa))
        self.assertEqual([p.incumbent for p in aa], sorted(self.configs))

    def test_delta(self):
        r = self.real
        self.assertAlmostEqual(delta(self.configs, r[0]), 0.06, places=12)
        self.assertAlmostEqual(delta(self.configs, r[1]), 0.005, places=12)
        self.assertAlmostEqual(delta(self.configs, r[36]), 80 / 456, places=12)
        self.assertAlmostEqual(delta(self.configs, r[37]), -80 / 456, places=12)
        self.assertEqual(sum(delta(self.configs, p) > 0 for p in r), 26)

    def test_recorded_decisions(self):
        self.assertEqual([recorded_decision(self.configs, p) for p in self.real], RECORDED)


class Loading(unittest.TestCase):
    def test_load_from_another_folder(self):
        rows = [("m", "d", "default", t, j, int(t == "b"), 0.5, 0.25) for t in ("b", "a") for j in range(TRIALS)]
        with tempfile.TemporaryDirectory() as folder:
            configs = load(write_csv(folder, rows))
        x = configs[("m", "d", "default")]
        self.assertEqual(x.tasks, ("a", "b"))
        self.assertEqual(x.rewards.tolist(), [[0] * 4, [1] * 4])
        np.testing.assert_allclose(x.costs, 0.75)

    def test_missing_trial_is_an_error(self):
        rows = [("m", "d", "default", "a", j, 1, 0.1, 0.1) for j in range(TRIALS - 1)]
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            load(write_csv(folder, rows))

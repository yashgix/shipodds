"""Tests for the recorded-trial loader."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shipodds import DATA_DIR
from shipodds.data import TRIALS, Config, load

HEADER = ("model", "domain", "mode", "task_id", "trial", "reward", "agent_cost", "user_cost")


def write_runs(folder, rows):
    path = Path(folder) / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def trial_row(trial, task="t1"):
    return ["m", "d", "default", task, trial, 1, "1.0", "0.5"]


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load(DATA_DIR)

    def test_configs_and_runs(self):
        self.assertEqual(len(self.configs), 26)
        self.assertEqual(sum(config.rewards.size for config in self.configs.values()), 10832)

    def test_config_shapes_and_types(self):
        for config in self.configs.values():
            self.assertIsInstance(config, Config)
            self.assertEqual(config.rewards.shape, (len(config.tasks), TRIALS))
            self.assertEqual(config.costs.shape, (len(config.tasks), TRIALS))
            self.assertEqual(config.rewards.dtype.kind, "i")
            self.assertEqual(config.costs.dtype.kind, "f")
            self.assertEqual(len(set(config.tasks)), len(config.tasks))

    def test_costs_are_positive(self):
        for config in self.configs.values():
            self.assertTrue(np.all(config.costs > 0), msg=config.key)

    def test_user_cost_is_zero_without_a_user(self):
        with (Path(DATA_DIR) / "runs.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["mode"] in ("no-user", "no-user-op"):
                    self.assertEqual(float(row["user_cost"]), 0.0, msg=row)

    def test_telecom_task_lists_match(self):
        lists = {config.tasks for config in self.configs.values() if config.domain in ("telecom", "telecom-workflow")}
        self.assertEqual(len(lists), 1)

    def test_tasks_are_sorted_strings(self):
        for config in self.configs.values():
            self.assertEqual(config.tasks, tuple(sorted(config.tasks)), msg=config.key)

    def test_keys_are_sorted(self):
        self.assertEqual(list(self.configs), sorted(self.configs))
        for key, config in self.configs.items():
            self.assertEqual(config.key, key)

    def test_load_accepts_string_and_path(self):
        self.assertEqual(list(load(str(DATA_DIR))), list(self.configs))
        self.assertEqual(list(load(Path(DATA_DIR))), list(self.configs))


class ExactLoadTests(unittest.TestCase):
    def test_rewards_and_costs_land_at_task_trial_positions(self):
        rows = [
            ["m", "d", "default", "t2", 1, 0, "2.0", "0.5"],
            ["m", "d", "default", "t1", 0, 1, "1.0", "0.25"],
            ["m", "d", "default", "t2", 3, 1, "3.0", "0.75"],
            ["m", "d", "default", "t1", 2, 0, "4.0", "0.0"],
            ["m", "d", "default", "t1", 3, 1, "5.0", "0.5"],
            ["m", "d", "default", "t2", 0, 1, "6.0", "0.5"],
            ["m", "d", "default", "t1", 1, 0, "7.0", "0.25"],
            ["m", "d", "default", "t2", 2, 0, "8.0", "0.75"],
        ]
        with tempfile.TemporaryDirectory() as folder:
            write_runs(folder, rows)
            configs = load(folder)
        self.assertEqual(list(configs), [("m", "d", "default")])
        config = configs[("m", "d", "default")]
        self.assertEqual(config.tasks, ("t1", "t2"))
        np.testing.assert_array_equal(config.rewards, np.array([[1, 0, 0, 1], [1, 0, 0, 1]], dtype=np.int64))
        np.testing.assert_allclose(
            config.costs,
            np.array([[1.25, 7.25, 4.0, 5.5], [6.5, 2.5, 8.75, 3.75]]),
        )


class BadRunsTests(unittest.TestCase):
    def assert_invalid(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            write_runs(folder, rows)
            with self.assertRaises(ValueError):
                load(folder)

    def test_missing_trial(self):
        self.assert_invalid([trial_row(0), trial_row(1), trial_row(2)])

    def test_duplicate_trial(self):
        self.assert_invalid([trial_row(0), trial_row(1), trial_row(2), trial_row(2)])

    def test_out_of_range_trial(self):
        self.assert_invalid([trial_row(0), trial_row(1), trial_row(2), trial_row(4)])

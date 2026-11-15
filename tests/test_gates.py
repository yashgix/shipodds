"""Tests for the gate rules."""

import unittest

import numpy as np

from promote_or_hold.gates import (
    BOOTSTRAP_DRAWS,
    OBRIEN_FLEMING,
    POCOCK,
    ROUNDS,
    RULES,
    Z_95,
    bootstrap_gate,
    bootstrap_weights,
    cmh_gate,
    cmh_z,
    mean_gate,
    sequential_gate,
)


class ConstantTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(ROUNDS, 4)
        self.assertEqual(BOOTSTRAP_DRAWS, 1000)
        self.assertEqual(Z_95, 1.96)
        self.assertEqual(POCOCK, (2.361,) * 4)
        self.assertEqual(OBRIEN_FLEMING, (4.048, 2.862, 2.337, 2.024))
        self.assertEqual(
            RULES,
            (
                "mean_1",
                "mean_2",
                "mean_4",
                "bootstrap_1",
                "bootstrap_2",
                "bootstrap_4",
                "cmh_1",
                "cmh_2",
                "cmh_4",
                "sequential_pocock",
                "sequential_obf",
            ),
        )


class MeanGateTests(unittest.TestCase):
    def test_strict_inequality_and_ties(self):
        sa = np.array([[1, 1], [1, 1], [0, 1]])
        sb = np.array([[1, 2], [1, 1], [0, 1]])
        np.testing.assert_array_equal(mean_gate(sa, sb), np.array([1, -1, -1]))


class BootstrapTests(unittest.TestCase):
    def test_weights_rows_sum_and_are_reproducible(self):
        first = bootstrap_weights(5, np.random.default_rng(123), draws=10)
        second = bootstrap_weights(5, np.random.default_rng(123), draws=10)
        self.assertEqual(first.shape, (10, 5))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=1), np.full(10, 5))

    def test_gate_on_clear_and_null_differences(self):
        weights = bootstrap_weights(4, np.random.default_rng(0), draws=100)
        sa = np.zeros((1, 4), dtype=np.int64)
        sb = np.ones((1, 4), dtype=np.int64)
        np.testing.assert_array_equal(bootstrap_gate(sa, sb, weights), [1])
        np.testing.assert_array_equal(bootstrap_gate(sb, sa, weights), [-1])
        np.testing.assert_array_equal(bootstrap_gate(sa, sa, weights), [0])


class CmhTests(unittest.TestCase):
    def test_hand_worked_z(self):
        sa = np.array([[0, 1]])
        sb = np.array([[2, 1]])
        self.assertAlmostEqual(cmh_z(sa, sb, 2)[0], 1.224744871391589)

    def test_zero_variance_is_zero(self):
        sa = np.array([[0, 2]])
        sb = np.array([[0, 2]])
        self.assertEqual(cmh_z(sa, sb, 2).item(), 0.0)

    def test_gate_thresholds(self):
        sa = np.array([[0, 0]])
        sb = np.array([[4, 4]])
        np.testing.assert_array_equal(cmh_gate(sa, sb, 4), [1])
        np.testing.assert_array_equal(cmh_gate(sb, sa, 4), [-1])
        undecided = cmh_gate(np.array([[1, 1]]), np.array([[2, 2]]), 4)
        np.testing.assert_array_equal(undecided, [0])


class SequentialGateTests(unittest.TestCase):
    def test_early_stopping_and_finality(self):
        ya = np.zeros((3, 2, 4), dtype=np.int64)
        yb = np.zeros((3, 2, 4), dtype=np.int64)

        # Replicate 0 promotes at round 2; later rounds must not reopen it.
        yb[0, :, :2] = 1
        ya[0, :, 2:] = 1

        # Replicate 1 holds at round 2; later rounds must not reopen it.
        ya[1, :, :2] = 1
        yb[1, :, 2:] = 1

        # Replicate 2 never crosses and uses all rounds.
        decisions, rounds_used = sequential_gate(ya, yb, POCOCK)
        np.testing.assert_array_equal(decisions, np.array([1, -1, 0]))
        np.testing.assert_array_equal(rounds_used, np.array([2, 2, 4]))


if __name__ == "__main__":
    unittest.main()

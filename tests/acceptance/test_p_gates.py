import unittest

import numpy as np

from shipodds.gates import (
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


class Constants(unittest.TestCase):
    def test_rules_and_bounds(self):
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
        self.assertEqual((ROUNDS, BOOTSTRAP_DRAWS, Z_95), (4, 1000, 1.96))
        self.assertEqual(tuple(POCOCK), (2.361,) * 4)
        self.assertEqual(tuple(OBRIEN_FLEMING), (4.048, 2.862, 2.337, 2.024))


class Mean(unittest.TestCase):
    def test_promotes_only_when_strictly_higher(self):
        sa = np.array([[1, 0], [1, 1], [2, 2]])
        sb = np.array([[1, 1], [1, 1], [1, 2]])
        self.assertEqual(mean_gate(sa, sb).tolist(), [1, -1, -1])


class Bootstrap(unittest.TestCase):
    def test_weights(self):
        w = bootstrap_weights(7, np.random.default_rng(5))
        self.assertEqual(w.shape, (BOOTSTRAP_DRAWS, 7))
        self.assertTrue(np.all(w.sum(1) == 7))
        picks = np.random.default_rng(5).integers(0, 7, (BOOTSTRAP_DRAWS, 7))
        self.assertEqual(w.tolist(), [np.bincount(p, minlength=7).tolist() for p in picks])

    def test_uses_the_2_5_and_97_5_percent_order_statistics(self):
        w = np.arange(40).reshape(40, 1)
        one = np.array([[0]]), np.array([[1]])
        self.assertEqual(bootstrap_gate(*one, w).tolist(), [1])
        self.assertEqual(bootstrap_gate(one[1], one[0], w).tolist(), [-1])
        self.assertEqual(bootstrap_gate(one[0], one[0], w).tolist(), [0])
        sa, sb = np.array([[0, 1]]), np.array([[1, 0]])  # a draw's total is +1 for (1, 0) and -1 for (0, 1)
        for negative, expected in ((1, 1), (2, 0)):
            w = np.array([[0, 1]] * negative + [[1, 0]] * (40 - negative))
            self.assertEqual(bootstrap_gate(sa, sb, w).tolist(), [expected], negative)

    def test_pairs_tasks(self):
        w = bootstrap_weights(6, np.random.default_rng(1))
        sa = np.array([[0, 1, 0, 1, 0, 1], [3, 3, 3, 3, 3, 3], [0, 2, 0, 2, 0, 2]])
        sb = np.array([[1, 2, 1, 2, 1, 2], [0, 0, 0, 0, 0, 0], [2, 0, 2, 0, 2, 0]])
        self.assertEqual(bootstrap_gate(sa, sb, w).tolist(), [1, -1, 0])


class Cmh(unittest.TestCase):
    def test_hand_computed(self):
        z = cmh_z(np.array([[0, 0, 1]]), np.array([[1, 1, 1]]), 1)
        self.assertAlmostEqual(float(z[0]), 2**0.5, places=12)
        z = cmh_z(np.array([[0, 1]]), np.array([[2, 2]]), 2)
        # task 1: m=2, E=1, V=1/3; task 2: m=3, E=1.5, V=1/4
        self.assertAlmostEqual(float(z[0]), 1.5 / (1 / 3 + 1 / 4) ** 0.5, places=12)

    def test_no_difference_gives_zero(self):
        z = cmh_z(np.array([[1, 0], [4, 4]]), np.array([[1, 0], [4, 4]]), 4)
        self.assertEqual(z.tolist(), [0.0, 0.0])

    def test_swapping_arms_flips_the_sign(self):
        rng = np.random.default_rng(3)
        sa, sb = rng.integers(0, 3, (5, 9)), rng.integers(0, 3, (5, 9))
        np.testing.assert_allclose(cmh_z(sa, sb, 2), -cmh_z(sb, sa, 2), atol=1e-12)

    def test_two_sided_95_threshold(self):
        # d tasks where only the candidate succeeds, on one trial each: z = sqrt(d)
        sa = np.zeros((2, 6), int)
        sb = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
        self.assertEqual(cmh_gate(sa, sb, 1).tolist(), [1, 0])
        self.assertEqual(cmh_gate(sb, sa, 1).tolist(), [-1, 0])


class Sequential(unittest.TestCase):
    def test_stops_at_the_first_crossing(self):
        n = 10
        ya, yb = np.zeros((1, n, 4), int), np.ones((1, n, 4), int)
        d, used = sequential_gate(ya, yb, POCOCK)  # round 1: z = sqrt(10) = 3.16
        self.assertEqual((d.tolist(), used.tolist()), ([1], [1]))
        d, used = sequential_gate(ya, yb, OBRIEN_FLEMING)  # round 2: z = sqrt(30) = 5.48
        self.assertEqual((d.tolist(), used.tolist()), ([1], [2]))
        d, used = sequential_gate(yb, ya, OBRIEN_FLEMING)
        self.assertEqual((d.tolist(), used.tolist()), ([-1], [2]))

    def test_a_decision_is_final(self):
        ya = np.zeros((1, 30, 4), int)
        yb = np.zeros((1, 30, 4), int)
        yb[:, :, 0], ya[:, :, 1:] = 1, 1
        d, used = sequential_gate(ya, yb, OBRIEN_FLEMING)
        self.assertEqual((d.tolist(), used.tolist()), ([1], [1]))

    def test_inconclusive_uses_every_round(self):
        same = np.ones((2, 8, 4), int)
        same[1, :, ::2] = 0
        d, used = sequential_gate(same, same.copy(), POCOCK)
        self.assertEqual((d.tolist(), used.tolist()), ([0, 0], [4, 4]))

    def test_replicates_are_independent(self):
        ya = np.zeros((3, 10, 4), int)
        yb = np.zeros((3, 10, 4), int)
        yb[0], ya[2] = 1, 1
        d, used = sequential_gate(ya, yb, POCOCK)
        self.assertEqual((d.tolist(), used.tolist()), ([1, 0, -1], [1, 4, 1]))

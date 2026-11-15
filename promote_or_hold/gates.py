"""Gate rules for promotion decisions on recorded agent trials."""

import numpy as np

ROUNDS = 4
BOOTSTRAP_DRAWS = 1000
Z_95 = 1.96
POCOCK = (2.361,) * 4
OBRIEN_FLEMING = (4.048, 2.862, 2.337, 2.024)
RULES = (
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
)
# v2 rules (SPEC-v2.md). Futility rules may be selected. Reference rules are reported but never selected.
FUTILITY_LOOKS = (2, 3)
FUTILITY = {"obf_futility_0.0": 0.0, "obf_futility_0.5": 0.5, "obf_futility_1.0": 1.0}
BASELINE_RULES = ("always_hold", "naive_peek")
NAIVE_PEEK = (Z_95,) * ROUNDS
V2_RULES = (*FUTILITY, *BASELINE_RULES)
ALL_RULES = RULES + V2_RULES
SELECTABLE = RULES + tuple(FUTILITY)


def mean_gate(sa, sb):
    """Promote only when the candidate's total successes are strictly higher."""
    sa = np.asarray(sa)
    sb = np.asarray(sb)
    return np.where(sb.sum(axis=-1) > sa.sum(axis=-1), 1, -1).astype(np.int64)


def always_hold_gate(replicates):
    """Keep the incumbent every time: never promotes, never pays for a run."""
    return np.full(replicates, -1, dtype=np.int64)


def bootstrap_weights(n, rng, draws=BOOTSTRAP_DRAWS):
    """Return draw-by-task counts of a task-level bootstrap resample."""
    picks = rng.integers(0, n, (draws, n))
    return np.stack([np.bincount(row, minlength=n) for row in picks])


def bootstrap_gate(sa, sb, weights):
    """Paired task bootstrap: promote, hold, or wait for a clearer difference."""
    totals = (np.asarray(sb) - np.asarray(sa)) @ weights.T
    totals = np.sort(totals, axis=1)
    tail = weights.shape[0] // 40
    lo = totals[:, tail]
    hi = totals[:, -1 - tail]
    return np.where(lo > 0, 1, np.where(hi < 0, -1, 0)).astype(np.int64)


def cmh_z(sa, sb, k):
    """Cochran-Mantel-Haenszel z statistic stratified by task."""
    sa = np.asarray(sa, dtype=float)
    sb = np.asarray(sb, dtype=float)
    k = float(k)
    m = sa + sb
    n_total = 2.0 * k
    expected = k * m / n_total
    variance = k * k * m * (n_total - m) / (n_total * n_total * (n_total - 1.0))
    num = (sb - expected).sum(axis=-1)
    denom = np.sqrt(variance.sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = num / denom
    return np.where(denom > 0, z, 0.0)


def cmh_gate(sa, sb, k):
    """Promote or hold when the CMH z statistic crosses the 95% threshold."""
    z = cmh_z(sa, sb, k)
    return np.where(z >= Z_95, 1, np.where(z <= -Z_95, -1, 0)).astype(np.int64)


def sequential_gate(ya, yb, bounds, futility=-np.inf):
    """Group-sequential gate: decide at the first look that crosses a boundary.

    With a futility threshold, a replicate whose z is below it at a look in FUTILITY_LOOKS stops and holds.
    """
    ya = np.asarray(ya)
    yb = np.asarray(yb)
    replicates = ya.shape[0]
    rounds = len(bounds)
    decisions = np.zeros(replicates, dtype=np.int64)
    rounds_used = np.full(replicates, rounds, dtype=np.int64)
    undecided = np.ones(replicates, dtype=bool)

    for r in range(1, rounds + 1):
        sa = ya[:, :, :r].sum(axis=2)
        sb = yb[:, :, :r].sum(axis=2)
        z = cmh_z(sa, sb, r)
        promote = undecided & (z >= bounds[r - 1])
        hold = undecided & ((z <= -bounds[r - 1]) | ((r in FUTILITY_LOOKS) & (z < futility)))
        decisions[promote] = 1
        decisions[hold] = -1
        decided = promote | hold
        rounds_used[decided] = r
        undecided &= ~decided

    return decisions, rounds_used

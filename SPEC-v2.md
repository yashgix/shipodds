# promote-or-hold, v2 changes

The v1 specification (not included in this repository) stays as written. This file lists what v2 adds, why, and which v1 behaviour
it changes. Everything not named here behaves as in v1, and the v1 acceptance tests in `tests/acceptance/`
pass unchanged. `results/metrics.json` stays byte-identical to v1.

## Why
- The v1 headline read as "the sequential gate lowers false promotions at lower cost". The data support two separate
  claims instead. A proper test (CMH) instead of the mean gate lowers false promotions, and `cmh_1` already does it
  at a quarter of the cost. Sequential stopping instead of a fixed four-trial test lowers cost at the same error rate.
- v1 picked its rule on the same data it reported, with caps set after a prototype run. v2 checks the pick on held-out
  domains.
- Plain O'Brien-Fleming almost never stops early on an identical copy, so it saves nothing when a change does
  nothing. v2 adds a futility stop, chosen inside the folds.
- v1 had no floor (a gate that never promotes), no check that the sequential boundaries matter (a fixed test checked
  after every round), no noise-model sensitivity, no power by effect size and no check of how the pick moves with
  the caps. v2 adds each.

## Rules (`promote_or_hold/gates.py`, `promote_or_hold/simulate.py`)
- `RULES` is unchanged. v2 adds `FUTILITY = {"obf_futility_0.0": 0.0, "obf_futility_0.5": 0.5, "obf_futility_1.0":
  1.0}`, `BASELINE_RULES = ("always_hold", "naive_peek")`, `V2_RULES = (*FUTILITY, *BASELINE_RULES)`,
  `ALL_RULES = RULES + V2_RULES` and `SELECTABLE = RULES + tuple(FUTILITY)` (a prefix of `ALL_RULES`).
- `sequential_gate(ya, yb, bounds, futility=-inf)`: unchanged by default. With a threshold, an undecided replicate
  whose z is below it at a look in `FUTILITY_LOOKS = (2, 3)` stops and holds. Look 1 is skipped because one trial per
  task carries too little information. The stop only holds, so it cannot raise the false-promotion rate (a
  non-binding futility boundary). The grid spans "not ahead at all" (0) to "not ahead by two standard errors" (2).
  The first grid stopped at 1, which was picked everywhere. 1.5 and 2 were added after an independent review, and
  the picks did not move.
- `bootstrap_gate` casts counts to signed integers before subtracting, so unsigned inputs cannot wrap a loss into a
  gain.
- `obf_futility_<f>`: `OBRIEN_FLEMING` with futility `f`. `always_hold_gate(replicates)`: every decision `-1`, no
  runs, no cost. `naive_peek`: `sequential_gate` with `NAIVE_PEEK = (Z_95,) * ROUNDS`.
- `v2_outcomes(ya, yb, cost)` returns the v2 rules from the same draws as `outcomes` (common random numbers).
- `draw_smoothed(a, b, rng, replicates)`: each task's pass rate is `(successes + 0.5) / (TRIALS + 1)`, trials are
  fresh Bernoulli draws at that rate, and each trial costs the task's mean recorded cost for both arms. The smoothing
  multiplies every true difference by the same factor, so no pair changes group.
- `simulate(..., v2=False, smoothed=False)`: `v2=True` adds the v2 rules. `smoothed=True` uses `draw_smoothed` with
  `rng = np.random.default_rng([SEED, 2, index])`. The recorded world keeps v1's stream, so v1 numbers are unchanged.

## Metrics (`promote_or_hold/summary.py`, `promote_or_hold/robustness.py`, `promote_or_hold/run.py`)
- `summary.py` holds what v1 and v2 share: `ERROR_CAP`, `POWER_SLACK`, `recommend(metrics, error_cap, power_slack)`
  (v1 behaviour at the defaults), `rule_metrics`, `distinct_comparisons`. `run.py` imports `recommend` from it.
- `main` refuses an `out` named `metrics_v2.json`, before simulating, so the v2 file cannot overwrite the v1 one.
- `main` simulates every pair in both worlds and writes `metrics.json` (v1) and `metrics_v2.json` beside it, with:
  - `rules` (= `ALL_RULES`), `selectable`, `distinct_comparisons`, and per rule the v1 per-rule keys plus
    `aa_cost_share` (mean cost on identical copies over their full-evaluation cost), all pairs, recorded world.
  - `pick`: the v1 selection procedure over `SELECTABLE` on all pairs.
  - `folds`: one per domain in `DOMAINS` (telecom-workflow counts as telecom). They are exploratory post-selection
    checks, since the selection procedure and the futility grid were designed after looking at all domains. The pick is made on the other two
    domains' real and identical-copy pairs. Every rule is scored on the held-out domain's pairs. Each fold also
    records its pair, distinct-comparison and identical-copy counts.
  - `smoothed`: the same per-rule keys and `pick` in the smoothed world.
  - `power_by_difference`: better pairs binned by true difference at `DELTA_BINS = (0, 0.03, 0.06, 0.10, 0.20)`, with
    each bin's pair count and every rule's better-promoted rate.
  - `pick_grid`: the pick for every `ERROR_CAPS = (0.025, 0.05, 0.10)` and `POWER_SLACKS = (0, 0.02, 0.05, 0.10, 0.20)`.

## Generated README (`promote_or_hold/report.py`, `demo.py`)
- `demo.py --readme` reads both files. The results block opens with a generated headline (held-out folds first, then
  the two in-sample effects and the peeking check), a caveat paragraph (distinct comparisons, fold sizes, in-sample v1
  pick, and a flag when every fold picks the largest futility threshold), then tables: all pairs side by side (false
  promotions, power, false rejections, inconclusive, cost on real pairs and on identical copies), held-out folds,
  power by true difference, recorded against smoothed, and the pick grid. The v1 paragraph and tables follow.
- No number is typed into the README or docs. The README and docs carry no semicolons.

## Housekeeping
- `make test` runs the own tests and the v1 acceptance tests (`tests/acceptance/`, byte-identical copies).
- `make lint` passes (`ruff check` and `ruff format --check`). The Makefile takes `PYTHON=...`.
- `.github/workflows/ci.yml` runs lint, the data check and the tests, then checks that the experiment reproduces the
  committed `results/`, with read-only permissions.

## Not in this change
Intervals over comparisons, validation on fresh runs, caching the incumbent's runs and non-inferiority margins.

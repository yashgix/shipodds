# promote-or-hold, v2 changes

`SPEC.md` is the v1 specification and stays as written. This file lists what v2 adds, why, and which v1 behaviour
it changes. Everything not named here behaves as in `SPEC.md`, and the v1 acceptance tests in `tests/acceptance/`
pass unchanged.

## Why
The v1 headline read as "the sequential gate lowers false promotions at lower cost". The data support two separate
claims instead. A proper test (CMH) instead of the mean gate lowers false promotions, and `cmh_1` already does it at
a quarter of the cost. Sequential stopping instead of a fixed four-trial test lowers cost at the same error rate. v1
also had no floor (a gate that never promotes) and no check that the sequential boundaries matter (a fixed test
checked after every round). v2 adds both as reported reference rules and rewrites the headline around the two
effects.

## Reference rules (`promote_or_hold/gates.py`, `promote_or_hold/simulate.py`)
- `BASELINE_RULES = ("always_hold", "naive_peek")`. They are kept apart from `RULES`, which is unchanged, so the
  v1 rule list, metrics and recommendation stay exactly as they were. `recommend` never considers them.
- `always_hold_gate(replicates)`: every decision is `-1`. Runs and cost are 0.
- `NAIVE_PEEK = (Z_95,) * ROUNDS`. `naive_peek` is `sequential_gate` with these bounds: the fixed 1.96 CMH test
  checked after every round, stopping at the first crossing in either direction. Runs and cost are paid per round
  used, as for the other sequential rules.
- `baseline_outcomes(ya, yb, cost)` returns `{rule: (decisions, runs, usd)}` for `BASELINE_RULES` from the same draws
  as `outcomes`, so all rules are compared on common random numbers.
- `simulate(..., baselines=False)`: with `baselines=True` the summary also holds the reference rules. The draws, and
  so every v1 number, are identical either way.

## Metrics (`promote_or_hold/run.py`)
- `results/metrics.json` is unchanged: the same 33 keys and values as v1.
- `main` also writes `metrics_v2.json` beside `out` (same JSON format: sorted keys, indent 1, 6-place rounding), with:
  `rules` (= `BASELINE_RULES`), `distinct_comparisons` (real pairs counted once per unordered incumbent/candidate set,
  26 here), and for each reference rule the v1 per-rule keys: `{aa,worse,better}_{promote,hold,inconclusive}`,
  `mean_runs`, `mean_cost_usd` and `cost_share`.
- `rule_metrics(...)` computes those per-rule keys for any rule list. v1 and v2 share it.

## Generated README (`promote_or_hold/report.py`, `demo.py`)
- `demo.py --readme` reads `metrics.json` and `metrics_v2.json` from `--results`. `render_readme` and
  `results_block` take the v2 metrics as an argument.
- The results block now opens with:
  - a headline paragraph that names the two effects: `mean_1` vs `cmh_1` for false promotions at equal cost, and
    `sequential_obf` vs `cmh_4` for power and cost at equal error. It gives the `naive_peek` identical-copy rate, and
    says its better power and lower cost come from that looser error rate, and what safety costs in power.
  - a caveat paragraph: the pairs are only `distinct_comparisons` distinct comparisons, and the recommended rule was
    picked on the same data with caps set after a prototype run and no held-out check.
  - a side-by-side table for `always_hold`, `mean_1`, `mean_4`, `cmh_1`, `cmh_4`, `naive_peek`,
    `sequential_pocock`, `sequential_obf`: identical copy promoted and worse promoted (false promotions), better
    promoted (power), better held (false rejection), better inconclusive (incumbent kept), and share of a full
    evaluation's cost.
- The v1 paragraph, rule table and per-pair table follow unchanged.

## Housekeeping
- `make test` runs the own tests and the v1 acceptance tests (`tests/acceptance/`, byte-identical copies).
- `make lint` passes (`ruff check` and `ruff format --check`). The Makefile takes `PYTHON=...`.
- `.github/workflows/ci.yml` runs lint, the data check and the tests, then checks that the experiment reproduces the
  committed `results/`, with read-only permissions.

## Not in this change
A futility stop, a held-out (leave-one-domain-out) check, a noise-model sensitivity run and intervals over
comparisons are planned next and are not part of v2 yet.

# promote-or-hold

**How often a promotion gate ships a worse agent, and what a safer gate costs.**

A team that runs AI agents in production changes them all the time: a new model, a new prompt, a new policy. Each
change goes through a *promotion gate*: run the candidate and the current version (the incumbent) on an evaluation
suite, then promote the candidate or keep the incumbent. Agent runs are noisy — the same agent on the same task
passes on one trial and fails on the next — and every run costs money. A gate can therefore be wrong in two ways:
it promotes a candidate that is really worse, or it spends far more runs than the decision needed.

This project measures that trade-off on real recorded agent trials: the tau2-bench paper's published runs (four
models, three customer-service domains, policy and setup variants, four trials of every task, with each run's
reward and recorded dollar cost). Eleven gate rules are replayed on those runs, plus two reference rules (always
hold, and a fixed test checked after every round). For each rule the experiment reports how often it promotes an
identical copy of the agent, how often it promotes a worse candidate, how often it promotes a better one, how often
it holds or cannot decide on a better one, and what a decision costs in runs and dollars.

**Headline.** The result is two separate effects, and they are easy to mix up. A proper statistical test, not the
sequential design, is what stops false promotions: a one-trial test already does it at a quarter of the cost. The
sequential design is what saves money: it keeps the test's false-promotion rate and almost all of the power of a
full four-trial test, at a lower cost. Every safe rule pays for its safety by promoting fewer truly better
candidates. The numbers are generated at the top of [Results](#results), next to the caveats that bound them.

## Why it matters

Agent changes ship constantly — a new model version, a reworded prompt, a new tool policy — and each one is a bet
that the new version is not worse. Teams make that bet with an evaluation suite and some rule for reading it. The
rule is usually informal ("the new one scored higher"), and informal rules on noisy, expensive runs fail in both
directions: they promote regressions, and they burn budget on evaluations that could have stopped earlier.

Making the rule explicit turns both failure modes into measurable quantities. That is what this repository does: it
takes one decision rule at a time, replays it on trials that were actually recorded, and reports the error rates and
the cost.

## The data

`data/runs.csv` is a derived extract of the 26 result files in `data/tau2/results/final/` of
[tau2-bench](https://github.com/sierra-research/tau2-bench) at commit
`c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4`. tau2-bench is MIT-licensed; `data/tau2-bench-LICENSE` is the license
copy and `data/tau2-results-README.md` the upstream folder's README. `data/MANIFEST.json` records the commit, each
upstream file's sha256, size and git blob id, and the extract's own sha256 and row count.

The extract has one row per recorded run, 10,832 rows in all:
`model, domain, mode, task_id, trial, reward, agent_cost, user_cost`.

- `model`: claude-3.7-sonnet, gpt-4.1, gpt-4.1-mini, o4-mini (2025 snapshots).
- `domain`: `airline` (50 tasks), `retail` (114), `telecom` (114), and `telecom-workflow` (the same 114 telecom tasks
  under a workflow-style agent policy).
- `mode`: `default`, or one of the telecom ablations `no-user`, `op`, `no-user-op` (the agent works without the
  simulated user, and/or with an oracle plan). The gpt-4.1-mini files call the default setup `base`; the extract
  writes `default`.
- `trial` 0–3, `reward` 0 or 1, and `agent_cost` / `user_cost`: the recorded USD cost of the agent's and the user
  simulator's model calls at 2025 prices. `user_cost` is 0 in the `no-user` and `no-user-op` modes.

A *config* is one (model, domain, mode) triple: 26 of them, each with a fixed task list and four trials of every
task. Those four trials per task are the raw material for everything else here.

The upstream folder says these are the tau2-bench paper's June 2025 runs, and that the airline and retail task sets
were revised after they were made (telecom is unchanged). They are a historical record, not current leaderboard
numbers. This project uses them only as real examples of how much an agent's result varies from trial to trial.

Rebuild or check the extract with:

```bash
python tools/fetch_data.py --verify      # check data/runs.csv against data/MANIFEST.json, no network
python tools/fetch_data.py               # rebuild it (downloads ~600 MB into data/raw/, which is git-ignored)
python tools/fetch_data.py --source DIR  # rebuild from a local copy of the upstream files, no network
```

## How to run

Python 3.14 with NumPy; the `ml-py314` environment has both.

```bash
make test                                         # unit tests
make lint                                         # ruff check + format check
python experiment.py --out results/metrics.json   # replay every gate on every pair (~20 s)
python demo.py                                    # one-line summary of results/metrics.json
python demo.py --readme README.md --out README.md # render the results block below
make readme                                       # the same, via the Makefile
```

`experiment.py` also takes `--data DIR` (the folder holding `runs.csv`) and `--replicates N`. Two runs with the same
arguments write byte-identical files. It writes the v1 metrics to `--out` and the v2 additions (the reference rules
and the count of distinct comparisons, see `SPEC-v2.md`) to `metrics_v2.json` beside it. `demo.py` reads both from
`results/` (or `--results DIR`) and only ever prints or renders what they contain.

## Results

<!-- results:start -->
**Two separate effects.** A proper test instead of the mean gate is what lowers false promotions. With one trial per task and the same cost, mean_1 promotes an identical copy 43.9% of the time and cmh_1 2.4%. A sequential test instead of a fixed one is what lowers cost. sequential_obf promotes 69.2% of truly better candidates against 70.2% for cmh_4, at 71.5% of a full evaluation's cost instead of 100.0%. Both promote an identical copy about as often (2.4% and 2.4%). The sequential boundaries matter: checking the fixed test after every round (naive_peek) promotes an identical copy 6.1% of the time. Its better power (72.3%) and lower cost (53.7%) come from that looser error rate. Safety costs power: mean_1 promotes 88.8% of better candidates, sequential_obf 69.2%, and always_hold none.

The 52 pairs are only 26 distinct comparisons, each counted in both directions, and some configs appear in several comparisons, so the group rates are not 52 independent measurements. The recommended rule was picked on these same results, by a rule whose caps were set after a prototype run on the same data. It has not been checked on held-out data.

| Rule | Identical copy promoted | Worse promoted | Better promoted (power) | Better held (false rejection) | Better inconclusive (incumbent kept) | Share of full evaluation |
|---|---|---|---|---|---|---|
| always_hold | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| mean_1 | 43.9% | 7.5% | 88.8% | 11.2% | 0.0% | 25.0% |
| mean_4 | 47.1% | 3.3% | 95.9% | 4.1% | 0.0% | 100.0% |
| cmh_1 | 2.4% | 0.0% | 38.6% | 0.0% | 61.3% | 25.0% |
| cmh_4 | 2.4% | 0.0% | 70.2% | 0.0% | 29.8% | 100.0% |
| naive_peek | 6.1% | 0.1% | 72.3% | 0.1% | 27.6% | 53.7% |
| sequential_pocock | 2.2% | 0.0% | 64.3% | 0.0% | 35.7% | 61.0% |
| sequential_obf | 2.4% | 0.0% | 69.2% | 0.0% | 30.7% | 71.5% |

Across 52 candidate/incumbent pairs (26 configs, 10,832 recorded runs; 26 identical-copy pairs), each gate was replayed 2,000 times. A full 4-trial evaluation of both arms costs $131.04 on average. The full recorded evaluation promotes 19 pairs, holds 19, and cannot decide 14.

| Rule | Identical copy promoted | Worse promoted | Better promoted | Inconclusive | Runs per decision | Cost per decision | Share of full evaluation |
|---|---|---|---|---|---|---|---|
| mean_1 | 43.9% | 7.5% | 88.8% | 0.0% | 198 | $32.76 | 25.0% |
| mean_2 | 45.8% | 5.1% | 93.1% | 0.0% | 397 | $65.52 | 50.0% |
| mean_4 | 47.1% | 3.3% | 95.9% | 0.0% | 794 | $131.03 | 100.0% |
| bootstrap_1 | 2.3% | 0.0% | 38.1% | 62.1% | 198 | $32.76 | 25.0% |
| bootstrap_2 | 2.4% | 0.0% | 44.2% | 55.8% | 397 | $65.52 | 50.0% |
| bootstrap_4 | 2.6% | 0.0% | 48.1% | 51.8% | 794 | $131.03 | 100.0% |
| cmh_1 | 2.4% | 0.0% | 38.6% | 61.5% | 198 | $32.76 | 25.0% |
| cmh_2 | 2.4% | 0.0% | 55.1% | 44.9% | 397 | $65.52 | 50.0% |
| cmh_4 | 2.4% | 0.0% | 70.2% | 29.9% | 794 | $131.03 | 100.0% |
| sequential_pocock | 2.2% | 0.0% | 64.3% | 35.8% | 498 | $79.93 | 61.0% |
| **sequential_obf** (recommended) | 2.4% | 0.0% | 69.2% | 30.8% | 577 | $93.65 | 71.5% |

| Incumbent | Candidate | Tasks | True difference (pts) | Full recorded evaluation | mean_1 | cmh_4 | sequential_obf |
|---|---|---|---|---|---|---|---|
| gpt-4.1 telecom | gpt-4.1 telecom-workflow | 114 | +17.5 | promote | 100.0% | 100.0% | 100.0% |
| o4-mini telecom | o4-mini telecom-workflow | 114 | +17.3 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1 telecom (no-user) | gpt-4.1 telecom-workflow (no-user) | 114 | +16.9 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1 telecom-workflow (op) | gpt-4.1 telecom (op) | 114 | +16.2 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1 telecom | claude-3.7-sonnet telecom | 114 | +15.1 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1-mini retail | claude-3.7-sonnet retail | 114 | +12.7 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1 telecom | gpt-4.1-mini telecom | 114 | +9.6 | promote | 98.9% | 99.8% | 99.8% |
| claude-3.7-sonnet airline | o4-mini airline | 50 | +9.0 | promote | 93.7% | 76.2% | 74.3% |
| o4-mini telecom-workflow (no-user-op) | o4-mini telecom (no-user-op) | 114 | +8.6 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1-mini airline | o4-mini airline | 50 | +8.5 | promote | 90.4% | 68.2% | 65.1% |
| o4-mini telecom-workflow (op) | o4-mini telecom (op) | 114 | +8.3 | promote | 100.0% | 100.0% | 100.0% |
| gpt-4.1-mini retail | gpt-4.1 retail | 114 | +8.1 | promote | 98.0% | 97.7% | 97.3% |
| gpt-4.1 telecom | o4-mini telecom | 114 | +7.9 | promote | 98.8% | 98.6% | 98.5% |
| o4-mini retail | claude-3.7-sonnet retail | 114 | +7.2 | promote | 96.6% | 92.1% | 91.7% |
| o4-mini telecom | claude-3.7-sonnet telecom | 114 | +7.2 | promote | 97.0% | 95.2% | 94.6% |
| claude-3.7-sonnet airline | gpt-4.1 airline | 50 | +6.0 | inconclusive | 84.0% | 42.1% | 39.1% |
| gpt-4.1-mini airline | gpt-4.1 airline | 50 | +5.5 | inconclusive | 79.0% | 29.4% | 27.2% |
| gpt-4.1-mini retail | o4-mini retail | 114 | +5.5 | promote | 87.1% | 63.1% | 61.2% |
| gpt-4.1-mini telecom | claude-3.7-sonnet telecom | 114 | +5.5 | promote | 87.8% | 67.5% | 65.2% |
| o4-mini telecom (no-user) | o4-mini telecom-workflow (no-user) | 114 | +4.8 | promote | 92.6% | 83.6% | 82.1% |
| gpt-4.1 retail | claude-3.7-sonnet retail | 114 | +4.6 | promote | 88.7% | 57.7% | 55.1% |
| gpt-4.1 airline | o4-mini airline | 50 | +3.0 | inconclusive | 62.7% | 10.3% | 9.6% |
| o4-mini retail | gpt-4.1 retail | 114 | +2.6 | inconclusive | 73.0% | 15.1% | 13.8% |
| gpt-4.1 telecom (no-user-op) | gpt-4.1 telecom-workflow (no-user-op) | 114 | +2.6 | inconclusive | 72.2% | 17.6% | 16.4% |
| o4-mini telecom | gpt-4.1-mini telecom | 114 | +1.8 | inconclusive | 62.3% | 8.8% | 8.3% |
| claude-3.7-sonnet airline | gpt-4.1-mini airline | 50 | +0.5 | inconclusive | 46.7% | 1.3% | 1.1% |
| gpt-4.1-mini airline | claude-3.7-sonnet airline | 50 | -0.5 | inconclusive | 38.4% | 0.2% | 0.2% |
| gpt-4.1-mini telecom | o4-mini telecom | 114 | -1.8 | inconclusive | 28.3% | 0.1% | 0.1% |
| gpt-4.1 retail | o4-mini retail | 114 | -2.6 | inconclusive | 20.1% | 0.0% | 0.0% |
| gpt-4.1 telecom-workflow (no-user-op) | gpt-4.1 telecom (no-user-op) | 114 | -2.6 | inconclusive | 18.5% | 0.0% | 0.0% |
| o4-mini airline | gpt-4.1 airline | 50 | -3.0 | inconclusive | 24.5% | 0.0% | 0.0% |
| claude-3.7-sonnet retail | gpt-4.1 retail | 114 | -4.6 | hold | 8.4% | 0.0% | 0.0% |
| o4-mini telecom-workflow (no-user) | o4-mini telecom (no-user) | 114 | -4.8 | hold | 4.6% | 0.0% | 0.0% |
| o4-mini retail | gpt-4.1-mini retail | 114 | -5.5 | hold | 7.6% | 0.0% | 0.0% |
| claude-3.7-sonnet telecom | gpt-4.1-mini telecom | 114 | -5.5 | hold | 7.0% | 0.0% | 0.0% |
| gpt-4.1 airline | gpt-4.1-mini airline | 50 | -5.5 | inconclusive | 13.2% | 0.0% | 0.0% |
| gpt-4.1 airline | claude-3.7-sonnet airline | 50 | -6.0 | inconclusive | 9.3% | 0.0% | 0.0% |
| claude-3.7-sonnet retail | o4-mini retail | 114 | -7.2 | hold | 2.1% | 0.0% | 0.0% |
| claude-3.7-sonnet telecom | o4-mini telecom | 114 | -7.2 | hold | 1.9% | 0.0% | 0.0% |
| o4-mini telecom | gpt-4.1 telecom | 114 | -7.9 | hold | 0.5% | 0.0% | 0.0% |
| gpt-4.1 retail | gpt-4.1-mini retail | 114 | -8.1 | hold | 1.5% | 0.0% | 0.0% |
| o4-mini telecom (op) | o4-mini telecom-workflow (op) | 114 | -8.3 | hold | 0.0% | 0.0% | 0.0% |
| o4-mini airline | gpt-4.1-mini airline | 50 | -8.5 | hold | 4.6% | 0.0% | 0.0% |
| o4-mini telecom (no-user-op) | o4-mini telecom-workflow (no-user-op) | 114 | -8.6 | hold | 0.0% | 0.0% | 0.0% |
| o4-mini airline | claude-3.7-sonnet airline | 50 | -9.0 | hold | 3.1% | 0.0% | 0.0% |
| gpt-4.1-mini telecom | gpt-4.1 telecom | 114 | -9.6 | hold | 0.5% | 0.0% | 0.0% |
| claude-3.7-sonnet retail | gpt-4.1-mini retail | 114 | -12.7 | hold | 0.1% | 0.0% | 0.0% |
| claude-3.7-sonnet telecom | gpt-4.1 telecom | 114 | -15.1 | hold | 0.0% | 0.0% | 0.0% |
| gpt-4.1 telecom (op) | gpt-4.1 telecom-workflow (op) | 114 | -16.2 | hold | 0.0% | 0.0% | 0.0% |
| gpt-4.1 telecom-workflow (no-user) | gpt-4.1 telecom (no-user) | 114 | -16.9 | hold | 0.0% | 0.0% | 0.0% |
| o4-mini telecom-workflow | o4-mini telecom | 114 | -17.3 | hold | 0.0% | 0.0% | 0.0% |
| gpt-4.1 telecom-workflow | gpt-4.1 telecom | 114 | -17.5 | hold | 0.0% | 0.0% | 0.0% |
<!-- results:end -->

## What this shows

Read the first table as three questions about each rule. How often does it promote something that is not better
(identical copy, worse)? How often does it promote a real improvement (power)? And when it does not, does it
actively hold or just fail to decide? In practice both of the last two keep the incumbent. The last column puts
each rule's cost against a full four-trial evaluation of both arms.

Two comparisons carry the result, and each isolates one effect.
- **Test versus mean gate, at equal cost.** `mean_1` and `cmh_1` see exactly the same trials. The only difference is
  whether the rule accounts for noise. That alone moves the identical-copy rate from the "higher mean wins" level to
  about the test's nominal 2.5%.
- **Sequential versus fixed, at equal error.** `sequential_obf` and `cmh_4` use the same statistic and promote
  identical copies about equally often. The sequential rule stops early when the answer is clear, so it costs less
  and gives up a little power.

`naive_peek` shows why the sequential boundaries are needed. It is the fixed test checked after every round with no
correction, which is what a team does when it "just keeps an eye on the numbers". It stops early too, but its
false-promotion rate is well above the nominal level. `always_hold` is the floor: no false promotions, no power, no
cost. A gate is only worth running for the improvements it lets through.

The stratified and bootstrap rules test different questions ("better on this suite?" versus "better on new tasks
like these?"), so they disagree on which pairs they can decide. The per-pair table shows where.

The recommended row is picked by the experiment, not by hand. Among the v1 rules whose identical-copy and
worse-promoted rates both stay under 5%, it keeps those within 5 points of the best better-promoted rate and takes
the cheapest. The caps were fixed after a prototype run on the same data, and the pick is checked on no held-out
data, so read it as a defensible choice under a stated policy, not as an optimum. The reference rules are never
recommended.

## Prior art

None of the statistics here is new; the measurement on real agent trials is the contribution.

- **pass^k** (Yao et al., [arXiv 2406.12045](https://arxiv.org/abs/2406.12045)) and **tau2-bench** (Barres et al.,
  [arXiv 2506.07982](https://arxiv.org/abs/2506.07982)): the benchmarks, and the reliability framing this data comes
  from.
- **Adding Error Bars to Evals** (Miller, [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)): paired,
  task-clustered comparisons and error bars for model evaluations.
- **Cochran–Mantel–Haenszel** (Cochran 1954; Mantel and Haenszel 1959): the stratified test used here, one stratum
  per task.
- **Group-sequential designs** (Pocock 1977; O'Brien and Fleming 1979; Jennison and Turnbull, *Group Sequential
  Methods*, 2000): the boundaries that let a gate look after every round without inflating its error rate.
- **The "peeking" problem** in online A/B tests: why looking at the data repeatedly with a fixed-sample test breaks
  it, and why sequential methods are the standard answer.

## Limits

- **The recorded runs are the world.** A pair's true difference is what its four recorded trials say, and a
  simulated new trial is one of those four. Per-task pass rates therefore take only five values (0, ¼, ½, ¾, 1),
  and no new noise is invented. Real trial-to-trial noise may be larger than this world's. Every error rate here is
  relative to the recorded runs, not to the agents' true quality.
- **At most four trials per task.** No rule can buy more information than the data holds, and the sequential rules
  stop after four looks.
- **Only 26 distinct comparisons.** Every pair appears in both directions, so the "worse" and "better" groups mirror
  each other. Some configs also appear in several comparisons. The group rates are averages over these 26
  comparisons, not 52 independent measurements, and the README gives no interval for them.
- **The task bootstrap answers a broader question** (how the gate behaves on new tasks drawn from a task
  distribution) than the truth used to score it (this suite). That is why it is more often inconclusive. It is a
  property of the question, not a defect of the method.
- **The sequential boundaries assume equal information per look.** The identical-copy promotion rate is the
  empirical check that they are not too aggressive here, not a proof.
- **2025 models and prices, customer-service domains, the paper's June 2025 task sets.** Nothing here speaks to
  current agents.
- **The recommendation rule and its caps were chosen after a prototype run on the same data.** The metrics are
  recomputed from the recordings every time, but the policy that picks the row was not pre-registered and is not
  checked on held-out data.
- **"Never promotes a worse candidate" is false.** The statistical gates' worse-promotion rates round to 0.0% in
  the table but are mostly not zero. They are small, not absent.

## Layout

```
promote_or_hold/   data loading, gate rules, simulation, experiment, report rendering
experiment.py      replays every gate on every pair and writes results/metrics.json and results/metrics_v2.json
demo.py            one-line summary, or renders the results block in this README
tests/             unit tests (unittest), with the v1 acceptance tests unchanged in tests/acceptance/
SPEC.md            the v1 specification
SPEC-v2.md         what v2 adds and why
data/              the pinned extract, its license copy and manifest
docs/              DESIGN.md and INTERVIEW.md
```

# shipodds

**How often a promotion gate ships a worse agent, and what a safer gate costs.**

Write-up with charts: <https://yashgix.github.io/shipodds/>

A team that runs AI agents in production changes them all the time: a new model, a new prompt, a new policy. Each
change goes through a *promotion gate*. It runs the candidate and the current version (the incumbent) on an
evaluation suite, then promotes the candidate or keeps the incumbent. Agent runs are noisy. The same agent on the
same task passes on one trial and fails on the next, and every run costs money. So a gate can be wrong in two ways.
It promotes a candidate that is really worse, or it spends far more runs than the decision needed.

This project measures that trade-off on real recorded agent trials: the tau2-bench paper's published runs (four
models, three customer-service domains, policy and setup variants, four trials of every task, with each run's
reward and recorded dollar cost). Eleven v1 gate rules are replayed on those runs. v2 adds five sequential rules
with a futility stop, one per threshold, and two reference rules (always hold, and a fixed test checked after every
round). For each rule the experiment reports how often it promotes an identical copy of the agent, a worse
candidate and a better one, how often it holds or cannot decide on a better one, and what a decision costs in runs
and dollars.

**Headline.** The result is two separate effects, and they are easy to mix up.
- A proper statistical test, not the sequential design, is what stops false promotions. A one-trial test already
  does it, at a quarter of the cost of a full evaluation.
- The sequential design is what saves money. With a futility stop, it keeps the test's false-promotion rate and
  nearly all the power of a full four-trial test at a clearly lower cost, also when the change does nothing.
- It also holds in an exploratory check that picks the rule on two domains and scores it on the third, in each of
  the three folds. The selection procedure and the futility thresholds were designed after looking at all domains,
  and the folds are small, so this is a post-selection consistency check, not an independent validation.
- Every safe rule pays for its safety by promoting fewer truly better candidates, mostly the ones with small gains.

The numbers are at the top of [Results](#results), next to the caveats that bound them.

## Why it matters

Agent changes ship constantly: a new model version, a reworded prompt, a new tool policy. Each one is a bet that the
new version is not worse. Teams make that bet with an evaluation suite and some rule for reading it. The rule is
usually informal ("the new one scored higher"), and informal rules on noisy, expensive runs fail in both
directions. They promote regressions, and they burn budget on evaluations that could have stopped earlier.

Making the rule explicit turns both failure modes into measurable quantities. That is what this repository does. It
takes one decision rule at a time, replays it on trials that were actually recorded, and reports the error rates and
the cost.

## A short example

```bash
$ python demo.py
mean_1 promotes an identical copy 43.9% of the time; the v1 pick (sequential_obf) 2.4%, at 71.5% of a full evaluation's cost.
```

An identical copy of an agent is never better, so every promotion of one is a mistake. The "higher mean wins" gate
with one trial per task makes that mistake in 44 of 100 evaluations. A stratified test at the same cost makes it in 2.

## The data

`data/runs.csv` is a derived extract of the 26 result files in `data/tau2/results/final/` of
[tau2-bench](https://github.com/sierra-research/tau2-bench) at commit
`c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4`. tau2-bench is MIT-licensed. `data/tau2-bench-LICENSE` is the license
copy and `data/tau2-results-README.md` the upstream folder's README. `data/MANIFEST.json` records the commit, each
upstream file's sha256, size and git blob id, and the extract's own sha256 and row count.

The extract has one row per recorded run, 10,832 rows in all:
`model, domain, mode, task_id, trial, reward, agent_cost, user_cost`.

- `model`: claude-3.7-sonnet, gpt-4.1, gpt-4.1-mini, o4-mini (2025 snapshots).
- `domain`: `airline` (50 tasks), `retail` (114), `telecom` (114), and `telecom-workflow` (the same 114 telecom tasks
  under a workflow-style agent policy).
- `mode`: `default`, or one of the telecom ablations `no-user`, `op`, `no-user-op` (the agent works without the
  simulated user, with an oracle plan, or both). The gpt-4.1-mini files call the default setup `base`, and the
  extract writes `default`.
- `trial` 0 to 3, `reward` 0 or 1, and `agent_cost` / `user_cost`: the recorded USD cost of the agent's and the user
  simulator's model calls at 2025 prices. `user_cost` is 0 in the `no-user` and `no-user-op` modes.

A *config* is one (model, domain, mode) triple. There are 26, each with a fixed task list and four trials of every
task. Those four trials per task are the raw material for everything else here.

The upstream folder says these are the tau2-bench paper's June 2025 runs, and that the airline and retail task sets
were revised after they were made (telecom is unchanged). They are a historical record, not current leaderboard
numbers. This project uses them only as real examples of how much an agent's result varies from trial to trial.

Rebuild or check the extract with:

```bash
python tools/fetch_data.py --verify      # check data/runs.csv against data/MANIFEST.json, no network
python tools/fetch_data.py               # rebuild it (downloads about 600 MB into data/raw/, which is git-ignored)
python tools/fetch_data.py --source DIR  # rebuild from a local copy of the upstream files, no network
```

## How to run

Python 3.14 and NumPy 2.5.4 (`pip install numpy==2.5.4 ruff`).

```bash
make test                                         # own tests and the v1 acceptance tests
make lint                                         # ruff check + format check
make experiment                                   # rerun the experiment, rewrite results/metrics.json
python experiment.py --out results/metrics.json   # replay every gate on every pair, in both worlds
python demo.py                                    # one-line summary of results/metrics.json
python demo.py --readme README.md --out README.md # render the results block below
make readme                                       # the same, via the Makefile
```

`experiment.py` also takes `--data DIR` (the folder holding `runs.csv`) and `--replicates N`. Two runs with the same
arguments write byte-identical files. It writes the v1 metrics to `--out` and everything v2 adds (see `SPEC-v2.md`)
to `metrics_v2.json` beside it. `demo.py` reads both from `results/` (or `--results DIR`) and only ever prints or
renders what they contain.

## Results

<!-- results:start -->
**Held out by domain (exploratory).** The selection procedure and the futility thresholds were designed after looking at all domains, so this is a post-selection check, not an independent validation. Picking the rule on two domains and scoring it on the third, the v1 selection procedure picks obf_futility_1.0 in each of the 3 folds. On the held-out domain the pick promotes an identical copy 1.8% to 2.1% of the time (cmh_4: 2.3% to 2.6%). It promotes 33.1% to 81.9% of better candidates against 37.9% to 83.6% for the fixed four-trial cmh_4, at most 4.9 points fewer. It costs 54.9% to 60.8% of a full evaluation on real pairs and 55.9% to 56.6% on identical copies. The one-shot cmh_1, at a quarter of the cost, promotes only 5.2% to 58.1% of better candidates.

**Two separate effects** (all pairs, in sample). A proper test instead of the mean gate is what lowers false promotions. With one trial per task and the same cost, mean_1 promotes an identical copy 43.9% of the time and cmh_1 2.4%. A sequential test instead of a fixed one is what lowers cost. sequential_obf promotes 69.2% of better candidates against 70.2% for cmh_4, at 71.5% of the cost, but 99.4% on identical copies, where it rarely stops early. Futility stops target that case. Checking the fixed test after every round (naive_peek) promotes an identical copy 6.1% of the time. Its power (72.3%) and cost (53.7%) come from that looser error rate.

**How small this is.** The 52 pairs are only 26 distinct comparisons, each counted in both directions, and some configs appear in several comparisons. Each held-out domain has few distinct comparisons and identical-copy pairs (airline 6 and 4, retail 6 and 4, telecom 14 and 18). There are only 3 folds, and no interval is given. The v1 recommended rule was picked on all the data, with caps set after a prototype run on the same data.

**All pairs, every rule that matters for the headline.**

| Rule | Identical copy promoted | Worse promoted | Better promoted (power) | Better held (false rejection) | Better inconclusive (incumbent kept) | Cost, real pairs | Cost, identical copies |
|---|---|---|---|---|---|---|---|
| always_hold | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| mean_1 | 43.9% | 7.5% | 88.8% | 11.2% | 0.0% | 25.0% | 25.0% |
| mean_4 | 47.1% | 3.3% | 95.9% | 4.1% | 0.0% | 100.0% | 100.0% |
| cmh_1 | 2.4% | 0.0% | 38.6% | 0.0% | 61.3% | 25.0% | 25.0% |
| cmh_4 | 2.4% | 0.0% | 70.2% | 0.0% | 29.8% | 100.0% | 100.0% |
| naive_peek | 6.1% | 0.1% | 72.3% | 0.1% | 27.6% | 53.7% | 94.0% |
| sequential_pocock | 2.2% | 0.0% | 64.3% | 0.0% | 35.7% | 61.0% | 98.0% |
| sequential_obf | 2.4% | 0.0% | 69.2% | 0.0% | 30.7% | 71.5% | 99.4% |
| obf_futility_0.0 | 2.4% | 0.0% | 69.2% | 6.3% | 24.5% | 60.3% | 74.1% |
| obf_futility_0.5 | 2.3% | 0.0% | 68.9% | 14.2% | 16.9% | 58.4% | 63.2% |
| obf_futility_1.0 | 2.0% | 0.0% | 67.1% | 24.0% | 8.9% | 56.4% | 56.2% |
| obf_futility_1.5 | 1.4% | 0.0% | 61.7% | 35.6% | 2.6% | 54.0% | 52.2% |
| obf_futility_2.0 | 0.8% | 0.0% | 52.6% | 47.1% | 0.3% | 51.7% | 50.6% |

**Held out by domain.** Picked on the other two domains, scored on this one.

| Held out | Rule | Identical copy promoted | Worse promoted | Better promoted (power) | Better held (false rejection) | Cost, real pairs | Cost, identical copies |
|---|---|---|---|---|---|---|---|
| airline | obf_futility_1.0 (pick) | 1.8% | 0.0% | 33.1% | 50.1% | 60.8% | 56.6% |
| airline | sequential_obf | 2.3% | 0.0% | 36.1% | 0.1% | 95.0% | 99.3% |
| airline | cmh_4 | 2.3% | 0.0% | 37.9% | 0.1% | 100.0% | 100.0% |
| airline | cmh_1 | 2.2% | 0.0% | 5.2% | 0.0% | 25.0% | 25.0% |
| retail | obf_futility_1.0 (pick) | 2.1% | 0.0% | 66.8% | 22.5% | 59.7% | 55.9% |
| retail | sequential_obf | 2.5% | 0.0% | 69.8% | 0.0% | 78.2% | 99.3% |
| retail | cmh_4 | 2.6% | 0.0% | 71.0% | 0.0% | 100.0% | 100.0% |
| retail | cmh_1 | 2.7% | 0.0% | 26.6% | 0.0% | 25.0% | 25.0% |
| telecom | obf_futility_1.0 (pick) | 2.0% | 0.0% | 81.9% | 13.5% | 54.9% | 56.2% |
| telecom | sequential_obf | 2.4% | 0.0% | 83.2% | 0.0% | 66.4% | 99.4% |
| telecom | cmh_4 | 2.4% | 0.0% | 83.6% | 0.0% | 100.0% | 100.0% |
| telecom | cmh_1 | 2.4% | 0.0% | 58.1% | 0.0% | 25.0% | 25.0% |

**Power by true difference** (better pairs, all domains).

| Rule | +0 to 3 pts (5 pairs) | +3 to 6 pts (5 pairs) | +6 to 10 pts (10 pairs) | +10 to 20 pts (6 pairs) |
|---|---|---|---|---|
| mean_1 | 63.4% | 87.0% | 95.7% | 100.0% |
| cmh_1 | 1.8% | 12.2% | 37.4% | 93.5% |
| cmh_4 | 10.6% | 60.3% | 87.0% | 100.0% |
| sequential_obf | 9.8% | 58.1% | 86.0% | 100.0% |
| obf_futility_1.0 | 8.6% | 53.5% | 83.4% | 100.0% |

**Smoothed-noise world.** Each task's pass rate is (successes + 0.5) / 5 and trials are drawn fresh.

| Rule | Identical copy promoted, recorded / smoothed | Worse promoted, recorded / smoothed | Better promoted (power), recorded / smoothed | Cost, real pairs, recorded / smoothed |
|---|---|---|---|---|
| mean_1 | 43.9% / 46.2% | 7.5% / 14.3% | 88.8% / 81.2% | 25.0% / 25.0% |
| cmh_1 | 2.4% / 2.5% | 0.0% / 0.2% | 38.6% / 23.3% | 25.0% / 25.0% |
| cmh_4 | 2.4% / 2.6% | 0.0% / 0.1% | 70.2% / 53.2% | 100.0% / 100.0% |
| sequential_obf | 2.4% / 2.6% | 0.0% / 0.1% | 69.2% / 52.3% | 71.5% / 81.8% |
| obf_futility_1.0 | 2.0% / 2.2% | 0.0% / 0.1% | 67.1% / 49.4% | 56.4% / 58.7% |

The smoothed world picks obf_futility_1.0.

**How the pick moves with the caps** (all domains).

| Error cap | slack 0 pts | slack 2 pts | slack 5 pts | slack 10 pts | slack 20 pts |
|---|---|---|---|---|---|
| 2.5% | cmh_4 | obf_futility_0.5 | obf_futility_1.0 | obf_futility_1.5 | cmh_2 |
| 5.0% | cmh_4 | obf_futility_0.5 | obf_futility_1.0 | obf_futility_1.5 | cmh_2 |
| 10.0% | cmh_4 | obf_futility_0.5 | obf_futility_1.0 | obf_futility_1.5 | cmh_2 |

**v1 results.**

Across 52 candidate/incumbent pairs (26 configs, 10,832 recorded runs, and 26 identical-copy pairs), each gate was replayed 2,000 times. A full 4-trial evaluation of both arms costs $131.04 on average. The full recorded evaluation promotes 19 pairs, holds 19, and cannot decide 14.

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
actively hold or just fail to decide? In practice both of the last two keep the incumbent. The cost columns put each
rule against a full four-trial evaluation of both arms, once over the real pairs and once over identical copies.

Two comparisons carry the result, and each isolates one effect.
- **Test versus mean gate, at equal cost.** `mean_1` and `cmh_1` see exactly the same trials. The only difference is
  whether the rule accounts for noise. That alone brings the identical-copy rate down to near the test's nominal
  level.
- **Sequential versus fixed, at equal error.** `sequential_obf` and `cmh_4` use the same statistic and promote
  identical copies about equally often. The sequential rule stops early when the answer is clear, so it costs less
  and gives up a little power. It almost never stops early when the change does nothing, which is the common case
  for a small prompt edit. The futility rules (`obf_futility_*`) also stop and hold at the second or third look when
  the candidate is not ahead by the threshold, which is where the extra saving comes from.

The held-out table runs the selection procedure on two domains and scores its pick on the third. The held-out
domain's pairs are left out of the pick itself. But the procedure, its caps and the futility thresholds were all
chosen after looking at every domain, so the table shows the pick is consistent across domains, not that it was
validated on unseen data. That would need fresh runs and a procedure frozen beforehand.

`naive_peek` shows why the sequential boundaries are needed. It is the fixed test checked after every round with no
correction, which is what a team does when it "just keeps an eye on the numbers". It stops early too, but its
false-promotion rate is well above the nominal level. `always_hold` is the floor: no false promotions, no power, no
cost. A gate is only worth running for the improvements it lets through.

The power-by-difference table shows where the safe rules are blind. They rarely promote a gain of a few points,
which a "higher mean wins" rule promotes often (along with many identical copies). The smoothed-noise world asks
whether the ranking survives noisier trials than the four recorded ones allow. The grid shows how the pick moves with
the power slack and the error cap.

The stratified and bootstrap rules test different questions ("better on this suite?" versus "better on new tasks
like these?"), so they disagree on which pairs they can decide. The per-pair table shows where.

The selection procedure is the v1 one. Among the rules whose identical-copy and worse-promoted rates both stay under
the error cap, it keeps those within the power slack of the best better-promoted rate and takes the cheapest. v1
applies it to the v1 rules on all the data (the recommended row in the v1 table). v2 applies it to the v1 rules plus
the futility rules, inside each fold. The reference rules are never picked.

## Prior art

None of the statistics here is new. The measurement on real agent trials is the contribution.

- **pass^k** (Yao et al., [arXiv 2406.12045](https://arxiv.org/abs/2406.12045)) and **tau2-bench** (Barres et al.,
  [arXiv 2506.07982](https://arxiv.org/abs/2506.07982)): the benchmarks, and the reliability framing this data comes
  from.
- **Adding Error Bars to Evals** (Miller, [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)): paired,
  task-clustered comparisons and error bars for model evaluations.
- **Cochran-Mantel-Haenszel** (Cochran 1954, Mantel and Haenszel 1959): the stratified test used here, one stratum
  per task.
- **Group-sequential designs** (Pocock 1977, O'Brien and Fleming 1979, Jennison and Turnbull, *Group Sequential
  Methods*, 2000): the boundaries that let a gate look after every round without inflating its error rate, and
  non-binding futility stops, which can only lower it.
- **The "peeking" problem** in online A/B tests: why looking at the data repeatedly with a fixed-sample test breaks
  it, and why sequential methods are the standard answer.

## Limits

- **The recorded runs are the world.** A pair's true difference is what its four recorded trials say, and a
  simulated new trial is one of those four. Per-task pass rates therefore take only five values, and no new noise is
  invented. Real trial-to-trial noise may be larger. The smoothed-noise world is one check of that, not a model of
  real agents. Every error rate here is relative to the simulated world, not to the agents' true quality.
- **At most four trials per task.** No rule can buy more information than the data holds, and the sequential rules
  stop after four looks.
- **Few distinct comparisons.** Every pair appears in both directions, so the "worse" and "better" groups mirror each
  other, and some configs appear in several comparisons. The group rates are averages over a few dozen comparisons,
  not independent measurements, and no interval is given.
- **Three small, exploratory folds.** Each held-out domain has only a handful of comparisons, and the selection
  design was fixed after looking at all of them. The folds are a consistency check, not an independent validation,
  and they cannot pin the numbers down.
- **The futility grid was widened after the first results.** The first grid stopped at the threshold that was
  picked, so larger thresholds were untested. The two larger thresholds were added afterwards. The picks did not
  move, because the larger thresholds lose more power than the slack allows.
- **The task bootstrap answers a broader question** (how the gate behaves on new tasks drawn from a task
  distribution) than the truth used to score it (this suite). That is why it is more often inconclusive. It is a
  property of the question, not a defect of the method.
- **The sequential boundaries assume equal information per look.** The identical-copy promotion rate is the
  empirical check that they are not too aggressive here, not a proof.
- **2025 models and prices, customer-service domains, the paper's June 2025 task sets.** Nothing here speaks to
  current agents.
- **The caps were chosen after a prototype run on the same data.** The v1 recommendation was not pre-registered. The
  held-out folds and the grid are the checks on that.
- **"Never promotes a worse candidate" is false.** The statistical gates' worse-promotion rates round to zero in the
  tables but are mostly not zero. They are small, not absent.

## Layout

```
shipodds/          data loading, gate rules, simulation, metrics, v2 checks, experiment, report rendering
tools/             fetch_data.py rebuilds the data extract, build_page.py renders docs/index.html from results/
experiment.py      replays every gate on every pair and writes results/metrics.json and results/metrics_v2.json
demo.py            one-line summary, or renders the results block in this README
tests/             unit tests (unittest), with the v1 acceptance tests unchanged in tests/acceptance/
SPEC-v2.md         what v2 adds and why (the v1 specification is not included)
data/              the pinned extract, its license copy and manifest
docs/              DESIGN.md (design notes) and index.html (the write-up page)
```

## Data and licenses

- **Our code:** MIT, see `LICENSE`.
- **tau2-bench results** (Sierra Research), MIT: `data/runs.csv` is a derived numeric extract of the paper's recorded
  runs (reward and cost per run, no conversation text). The upstream license is kept as `data/tau2-bench-LICENSE` and
  the upstream folder's README as `data/tau2-results-README.md`. The original result files are not stored here.
  `tools/fetch_data.py` rebuilds the extract from the pinned upstream commit and checks every file against the
  sha256 sums in `data/MANIFEST.json`.
- **NumPy** (BSD-3-Clause) and **ruff** (MIT) are installed by the user and not shipped.

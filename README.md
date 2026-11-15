# promote-or-hold

**How often a promotion gate ships a worse agent, and what a safer gate costs.**

A team that runs AI agents in production changes them all the time: a new model, a new prompt, a new policy. Each
change goes through a *promotion gate*: run the candidate and the current version (the incumbent) on an evaluation
suite, then promote the candidate or keep the incumbent. Agent runs are noisy — the same agent on the same task
passes on one trial and fails on the next — and every run costs money. A gate can therefore be wrong in two ways:
it promotes a candidate that is really worse, or it spends far more runs than the decision needed.

This project measures that trade-off on real recorded agent trials: the tau2-bench paper's published runs (four
models, three customer-service domains, policy and setup variants, four trials of every task, with each run's
reward and recorded dollar cost). Eleven gate rules are replayed on those runs. For each rule the experiment reports
how often it promotes an identical copy of the agent, how often it promotes a worse candidate, how often it
promotes a better one, how often it cannot decide, and what a decision costs in runs and dollars.

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

`experiment.py` also takes `--data DIR` (the folder holding `runs.csv`) and `--replicates N`; two runs with the same
arguments write byte-identical files. `demo.py` reads `results/metrics.json` (or `--results DIR`) and only ever
prints or renders what that file contains.

## Results

<!-- results:start -->
<!-- results:end -->

## What this shows

Read the table above as two costs that trade against each other. The identical-copy and worse-promoted columns are
the first cost: how often a gate promotes an agent that is not better, which is what a team feels when a rollout
regresses. The better-promoted and inconclusive columns are the second: how much of the time a gate refuses to act
on a real improvement. The last columns translate both into runs and dollars, measured against a full four-trial
evaluation of both arms.

The comparison across rows is the point of the exercise. The one-trial mean rule is what a team writes when it has
not thought about noise, and its identical-copy column shows the price of that. The stratified and bootstrap rules
test different questions — "better on this suite?" versus "better on new tasks like these?" — so they disagree on
which pairs they can decide, and the per-pair table shows where. Group-sequential versions answer the same question
as their fixed-trial counterparts but stop as soon as the answer is clear, which is why their cost column sits below
a full evaluation.

The recommended row is picked by the experiment, not by hand: among the rules whose identical-copy and
worse-promoted rates both stay under 5%, it keeps those within 5 points of the best better-promoted rate and takes
the cheapest. The caps themselves were fixed after a prototype run, so read the recommendation as a defensible
choice under a stated policy, not as an optimum. And none of these gates is perfect: a rule whose worse-promoted
rate rounds to zero in the table still promotes a worse candidate occasionally.

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
  and no new noise is invented. Real trial-to-trial noise may be larger than this world's; every error rate here is
  relative to the recorded runs, not to the agents' true quality.
- **At most four trials per task.** No rule can buy more information than the data holds, and the sequential rules
  stop after four looks.
- **Every pair appears in both directions**, so the "worse" and "better" groups mirror each other: 26 distinct
  comparisons, not 52 independent ones.
- **The task bootstrap answers a broader question** (how the gate behaves on new tasks drawn from a task
  distribution) than the truth used to score it (this suite). That is why it is more often inconclusive; it is a
  property of the question, not a defect of the method.
- **The sequential boundaries assume equal information per look.** The identical-copy promotion rate is the
  empirical check that they are not too aggressive here, not a proof.
- **2025 models and prices, customer-service domains, the paper's June 2025 task sets.** Nothing here speaks to
  current agents.
- **The recommendation rule and its caps were chosen after a prototype run.** The metrics are recomputed from the
  recordings every time; the policy that picks the row was not pre-registered.
- **"Never promotes a worse candidate" is false.** The statistical gates' worse-promotion rates round to 0.0% in
  the table but are mostly not zero; they are small, not absent.

## Layout

```
promote_or_hold/   data loading, gate rules, simulation, experiment, report rendering
experiment.py      replays every gate on every pair and writes results/metrics.json
demo.py            one-line summary, or renders the results block in this README
tests/             unit tests (unittest)
data/              the pinned extract, its license copy and manifest
docs/              DESIGN.md and INTERVIEW.md
```

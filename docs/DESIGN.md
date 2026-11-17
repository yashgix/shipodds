# Design notes

## The data, and what a config is

`data/runs.csv` holds one row per recorded agent run from the tau2-bench paper's June 2025 results: the model, the
domain, the setup mode, the task id, the trial index (0–3), the binary reward, and the USD cost of the agent's and
the user simulator's model calls. It is pinned by `data/MANIFEST.json` and rebuilt only by `tools/fetch_data.py`.

A **config** is one `(model, domain, mode)` triple. `shipodds.data.load` groups the rows into 26 configs,
each a `Config` holding the sorted tuple of task ids and two arrays shaped `(tasks, 4)`: `rewards[i, j]` is the
reward of trial `j` of task `i`, and `costs[i, j]` the summed cost of that trial. The loader rejects a task that
does not have each of trials 0–3 exactly once, so every later array operation can assume the full rectangle.

Grouping by task is what makes paired comparisons possible: incumbent and candidate were run on the same task, so
per-task differences are meaningful and far less noisy than differences between overall means. Trials within a task
are treated as exchangeable draws from that task's behaviour. That is the working assumption of the whole design.

## Pairs

Three kinds of comparison, all drawn from the same 26 configs:

- **Model swaps** (36): within each of airline, retail and telecom, every ordered pair of distinct models in the
  `default` mode. These are the changes a team actually considers, such as "should we move to the newer model?"
- **Policy changes** (16): the telecom task suite under the plain agent versus the workflow-style policy, in both
  directions, for the two OpenAI models. Same tasks, different scaffolding.
- **Identical copies** (26): each config compared with itself, candidate arm equal to incumbent arm. Nothing can be
  better here, so every promotion is a false promotion, and the rate is the cleanest measure of how trigger-happy a
  gate is.

A pair's **true difference** is `delta`: the candidate's mean reward over all its recorded runs minus the
incumbent's. Pairs with `delta <= 0` are "worse" (the gate should hold) and pairs with `delta > 0` are "better"
(the gate should promote). Because every comparison appears in both directions, the two groups mirror each other.
That is a deliberate consequence of using all ordered pairs, and it is a limit of the design, not 52 independent
comparisons.

## The simulation: the recorded runs are the world

No model is called. Instead each gate is replayed many times on each pair:

- A simulated trial of a task is one of that task's four recorded trials, drawn uniformly at random, carrying that
  trial's reward and cost. The two arms are drawn independently, so the same task can be "easy" for one arm in one
  replicate and hard for the other.
- `draw` fills `ya`, `yb` and `cost` shaped `(replicates, tasks, 4)`. The columns are the looks a sequential gate
  uses.
- A fixed rule with `k` trials per task and arm pays `2 · tasks · k` runs and the recorded cost of those runs. A
  sequential rule pays only for the rounds it used before deciding.

This is a trial-level bootstrap *within* each task, not across tasks. It answers "how would this gate have decided
if these agents had been run again on the same suite?" That is the question a promotion gate faces. A separate,
task-level bootstrap is one of the gate rules: it resamples tasks with replacement, and so answers a broader
question, "would the candidate be better on new tasks like these?".

The price of treating the recorded runs as the world is that per-task pass rates take only five values and trial
noise cannot exceed what was recorded. The error rates are therefore relative to this world. The README's limits
section spells that out.

## Seeds and reproducibility

Every random draw is seeded, so two runs of the experiment produce byte-identical metrics:

- `SEED = 20261114`.
- The trial draw for pair number `i` uses `np.random.default_rng([SEED, 1, i])`, with `i` the pair's position in the
  real-pair list and `52 + i` for identical-copy pair `i`. The index is all the simulation needs to reproduce a
  pair, which also keeps the pairs independent of the order in which they are run.
- The task-bootstrap weights for a suite of `n` tasks use `np.random.default_rng([SEED, 0, n])` and are cached: the
  same 1,000 resamples serve every pair with that many tasks, which is what keeps the bootstrap comparison paired
  across rules.
- `REPLICATES = 2000` per pair and rule. `BOOTSTRAP_DRAWS = 1000` resamples inside the bootstrap gate.

## The rules, and the question each answers

Decisions are coded `1` promote, `-1` hold, `0` inconclusive.

- `mean_k`: promote when the candidate's total successes over `k` trials are strictly higher, otherwise hold. Never
  inconclusive. This is the informal rule, included as a baseline rather than as a recommendation.
- `bootstrap_k`: resample tasks with replacement, take the 2.5th and 97.5th percentiles of the total paired
  difference, promote if the whole interval is above zero, hold if it is below, otherwise inconclusive. Question:
  *better on new tasks like these?*
- `cmh_k`: Cochran–Mantel–Haenszel, stratified by task, against a two-sided 5% threshold. Question: *better on this
  suite?* Stratifying by task means a few easy tasks cannot dominate the comparison.
- `sequential_pocock` and `sequential_obf`: the same CMH statistic recomputed after each round on cumulative
  successes, with Pocock or O'Brien–Fleming boundaries for four equally spaced looks. The first crossing decides,
  and a decision is final. Question: the CMH question, but willing to stop early when the answer is already clear.

The two sequential boundaries make different trade-offs: Pocock spends alpha evenly and decides earlier on
borderline cases, while O'Brien–Fleming is conservative early and only becomes demanding at the end.

## The recommendation rule

`recommend` is a policy, not a statistic. Among the eleven rules it keeps those whose identical-copy and
worse-candidate promotion rates are both at most 5%. Among those it keeps the rules whose better-candidate
promotion rate is within 5 points of the best kept rate. Then it returns the cheapest of them by mean cost per
decision, breaking ties by position in `RULES`. The two 5% numbers are a stated tolerance for the two kinds of
error, not a derived optimum, and they were fixed after a prototype run. The metrics themselves are recomputed from
the recordings every time. Only the policy is prior.

## Trade-offs and what is deliberately absent

- **Fixed trials versus sequential.** Fixed rules are simple and easy to explain. Sequential rules buy cost savings
  at the price of boundaries that assume looks are equally informative. Four rounds is all the data allows, so the
  savings are bounded.
- **This suite versus new tasks.** The CMH test is powerful when the recorded task suite is the population. The task
  bootstrap is honest about generalising beyond it and pays for that honesty in inconclusive decisions.
- **No new runs, no regrading.** Rewards are taken as given and no model is called. This project measures
  decision procedures, not agents.
- **Not here:** non-inferiority margins, cached-incumbent designs, adaptive task selection, Bayesian or
  anytime-valid tests, and comparisons across more than two arms. Each would change the error accounting and
  deserves its own measurement.

## v2 additions (see `SPEC-v2.md`)

- **Futility stops.** `obf_futility_<f>` is the O'Brien-Fleming gate that also stops and holds at look 2 or 3 when
  z is below `f`, for `f` in `FUTILITY`. Look 1 is skipped because one trial per task says too little. The stop only
  ever holds, so it can only lower the false-promotion rate. Its cost is power, and it shows up as more "held"
  decisions on better candidates. The threshold is chosen by the selection procedure, never fixed by hand.
- **Reference rules.** `always_hold` never promotes and costs nothing. `naive_peek` checks the fixed 1.96 test after
  every round. Both are reported and never selected.
- **Held out by domain (exploratory).** The selection procedure picks among the v1 rules and the futility rules on
  two domains and the pick is scored on the third. The procedure and the futility grid were designed after looking
  at all domains, so this is a consistency check, not an independent validation. Telecom and telecom-workflow
  count as one domain, so the policy comparisons stay in the telecom fold.
- **Smoothed-noise world.** Each task's pass rate becomes `(successes + 0.5) / (TRIALS + 1)` (the Jeffreys posterior
  mean) and trials are drawn fresh at that rate, at the task's mean recorded cost. Tasks that passed or failed all
  four recorded trials become noisy. The smoothing shrinks every true difference by the same factor, so no pair
  changes group. It uses its own random stream, so the recorded-world numbers are unchanged.
- **Power by true difference** bins the better pairs by their true difference, and **the pick grid** reruns the
  selection over a range of error caps and power slacks.

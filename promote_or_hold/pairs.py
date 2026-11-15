"""Candidate/incumbent pairs drawn from the recorded tau2-bench configs."""

import itertools
from dataclasses import dataclass

DOMAINS = ("airline", "retail", "telecom")
MODES = ("default", "no-user", "op", "no-user-op")
POLICY_MODELS = ("gpt-4.1-2025-04-14", "o4-mini-2025-04-16")


@dataclass(frozen=True)
class Pair:
    """A candidate config compared against an incumbent config."""

    kind: str
    incumbent: tuple[str, str, str]
    candidate: tuple[str, str, str]


def real_pairs(configs) -> list[Pair]:
    """Return the 52 real comparisons: 36 model swaps and 16 policy changes."""
    models = sorted({model for model, _, _ in configs})
    pairs = [
        Pair("model", (a, domain, "default"), (b, domain, "default"))
        for domain in DOMAINS
        for a, b in itertools.permutations(models, 2)
    ]
    for model in POLICY_MODELS:
        for mode in MODES:
            pairs.append(Pair("policy", (model, "telecom", mode), (model, "telecom-workflow", mode)))
            pairs.append(Pair("policy", (model, "telecom-workflow", mode), (model, "telecom", mode)))
    return pairs


def aa_pairs(configs) -> list[Pair]:
    """Return one identical-copy pair per config, sorted by key."""
    return [Pair("aa", key, key) for key in sorted(configs)]


def delta(configs, pair) -> float:
    """Candidate mean reward minus incumbent mean reward over all recorded runs."""
    incumbent = configs[pair.incumbent]
    candidate = configs[pair.candidate]
    return float(candidate.rewards.mean() - incumbent.rewards.mean())

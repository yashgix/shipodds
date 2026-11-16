"""Load tau2-bench's recorded trials into per-config arrays."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import DATA_DIR

TRIALS = 4


@dataclass(frozen=True, eq=False)
class Config:
    """One (model, domain, mode) setup and its recorded trials."""

    model: str
    domain: str
    mode: str
    tasks: tuple[str, ...]
    rewards: np.ndarray
    costs: np.ndarray

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model, self.domain, self.mode)


def load(data_dir=DATA_DIR) -> dict:
    """Read runs.csv and return configs keyed by (model, domain, mode), sorted."""
    path = Path(data_dir) / "runs.csv"
    rows: dict[tuple[str, str, str], dict[str, list[tuple[int, int, float]]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["domain"], row["mode"])
            trial = int(row["trial"])
            reward = int(row["reward"])
            cost = float(row["agent_cost"]) + float(row["user_cost"])
            rows.setdefault(key, {}).setdefault(row["task_id"], []).append((trial, reward, cost))

    configs = {}
    for key, by_task in rows.items():
        tasks = tuple(sorted(by_task))
        rewards = np.empty((len(tasks), TRIALS), dtype=np.int64)
        costs = np.empty((len(tasks), TRIALS), dtype=float)
        for i, task in enumerate(tasks):
            entries = by_task[task]
            if len(entries) != TRIALS:
                raise ValueError(f"{key} task {task!r} has {len(entries)} trials, expected {TRIALS}")
            seen = set()
            for trial, reward, cost in entries:
                if not 0 <= trial < TRIALS or trial in seen:
                    raise ValueError(f"{key} task {task!r} has an invalid or duplicate trial {trial}")
                seen.add(trial)
                rewards[i, trial] = reward
                costs[i, trial] = cost
            if len(seen) != TRIALS:
                raise ValueError(f"{key} task {task!r} is missing trials")
        configs[key] = Config(*key, tasks, rewards, costs)

    return {key: configs[key] for key in sorted(configs)}

"""Rebuild data/runs.csv from tau2-bench's published trial results and check it against data/MANIFEST.json.

The upstream files (26 JSON files, about 560 MB) are downloaded at the pinned commit into data/raw/, which git
ignores. `--source DIR` reads them from a local copy instead, and `--verify` only checks data/runs.csv; neither
uses the network.
"""

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SUFFIX = "_gpt-4.1-2025-04-14_4trials.json"
COLUMNS = ("model", "domain", "mode", "task_id", "trial", "reward", "agent_cost", "user_cost")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(manifest, raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name in manifest["upstream"]:
        target = raw / name
        if not target.is_file():
            url = f"{manifest['raw_base']}{manifest['commit']}/{manifest['path']}{name}"
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "fetch_data"}), timeout=300
            ) as r:
                target.write_bytes(r.read())


def check_upstream(manifest, raw):
    return [
        name
        for name, entry in manifest["upstream"].items()
        if not (raw / name).is_file() or sha256(raw / name) != entry["sha256"]
    ]


def rows(path):
    # "base" is the gpt-4.1-mini file's name for the default setup (same policy, user simulator and settings).
    model, domain, mode = path.name.removesuffix(SUFFIX).split("_")
    mode = "default" if mode == "base" else mode
    sims = json.loads(path.read_text())["simulations"]
    for s in sorted(sims, key=lambda s: (s["task_id"], s["trial"])):
        yield [
            model,
            domain,
            mode,
            s["task_id"],
            s["trial"],
            int(s["reward_info"]["reward"]),
            repr(s["agent_cost"]),
            repr(s["user_cost"]),
        ]


def build(manifest, raw):
    with open(DATA / "runs.csv", "w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(COLUMNS)
        for name in sorted(manifest["upstream"]):
            writer.writerows(rows(raw / name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check data/runs.csv only; no network")
    parser.add_argument("--source", type=Path, help="a local folder with the upstream files; no network")
    args = parser.parse_args()
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    if not args.verify:
        raw = args.source or DATA / "raw"
        if not args.source:
            download(manifest, raw)
        if bad := check_upstream(manifest, raw):
            sys.exit(f"upstream files missing or changed: {bad}")
        build(manifest, raw)
    if sha256(DATA / "runs.csv") != manifest["runs.csv"]["sha256"]:
        sys.exit("data/runs.csv does not match MANIFEST.json")
    print(f"data/runs.csv matches MANIFEST.json ({manifest['runs.csv']['rows']} runs)")


if __name__ == "__main__":
    main()

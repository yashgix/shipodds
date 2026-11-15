"""Command-line entry point for the promotion-gate experiment."""

import argparse
from pathlib import Path

from promote_or_hold import DATA_DIR
from promote_or_hold.run import main
from promote_or_hold.simulate import REPLICATES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="where to write metrics.json")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder containing runs.csv")
    parser.add_argument("--replicates", type=int, default=REPLICATES, help="simulated runs per pair")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.out, args.replicates)

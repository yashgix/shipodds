"""Print a one-line result summary or render README.md from metrics.json."""

import argparse
import json
from pathlib import Path

from promote_or_hold.data import load
from promote_or_hold.pairs import real_pairs
from promote_or_hold.report import render_readme, summary_line


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="directory containing metrics.json")
    parser.add_argument("--readme", type=Path, help="README template to render")
    parser.add_argument("--out", type=Path, help="where to write the rendered README")
    args = parser.parse_args(argv)
    if args.readme is not None and args.out is None:
        parser.error("--readme requires --out")
    return args


def main(argv=None):
    args = parse_args(argv)
    metrics = json.loads((args.results / "metrics.json").read_text())
    if args.readme is None:
        print(summary_line(metrics))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_readme(args.readme.read_text(), metrics, real_pairs(load())))


if __name__ == "__main__":
    main()

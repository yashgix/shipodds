.PHONY: test lint data-check experiment readme

PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	ruff check . && ruff format --check .

data-check:
	$(PYTHON) tools/fetch_data.py --verify

experiment:
	$(PYTHON) experiment.py --out results/metrics.json

readme:
	$(PYTHON) demo.py --readme README.md --out README.md

PY ?= python
CONFIG ?= configs/experiments.yaml
DATASETS ?= configs/datasets.yaml
OUT ?= results

.PHONY: install dev test lint synthetic quick download preprocess paper clean

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,io]"

test:
	$(PY) -m pytest

lint:
	ruff check probc tests --select E,F,W --ignore E501

# All experiments on synthetic stand-ins of the paper's datasets (no downloads)
synthetic:
	$(PY) -m probc run all --synthetic --config $(CONFIG) --out-dir $(OUT)/synthetic

# Fast smoke test
quick:
	$(PY) -m probc run all --synthetic --quick --config $(CONFIG) --out-dir $(OUT)/quick

# ---- real data -----------------------------------------------------------------
download:
	$(PY) -m probc download --datasets-config $(DATASETS)

preprocess:
	bash scripts/preprocess_all.sh

paper:
	$(PY) -m probc run all --config $(CONFIG) --out-dir $(OUT)/paper

clean:
	rm -rf $(OUT)/quick .pytest_cache

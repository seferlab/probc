#!/usr/bin/env bash
# End-to-end reproduction on the real datasets:
#   1) download genomes, ENCODE/Roadmap peaks and Hi-C/Micro-C files
#   2) build restriction-site level datasets
#   3) run every experiment (Figs. 1-13, Tables 2-3)
set -euo pipefail
python -m probc download --datasets-config configs/datasets.yaml
bash scripts/preprocess_all.sh
python -m probc run all --config configs/experiments.yaml --out-dir results/paper

#!/usr/bin/env bash
# Build and cache every processed dataset used by the experiments.
# Edit CHROMS to restrict to a subset (processing all chromosomes at
# restriction-site resolution needs a lot of memory and time).
set -euo pipefail
DATASETS=${DATASETS:-configs/datasets.yaml}
HUMAN_CHROMS=${HUMAN_CHROMS:-"chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX"}
MOUSE_CHROMS=${MOUSE_CHROMS:-"chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chrX"}

for ds in hESC IMR90; do
  python -m probc preprocess --datasets-config "$DATASETS" --dataset "$ds" --chroms $HUMAN_CHROMS
  for inc in "histone" "tf dnase"; do
    python -m probc preprocess --datasets-config "$DATASETS" --dataset "$ds" --chroms $HUMAN_CHROMS --include $inc
  done
done
python -m probc preprocess --datasets-config "$DATASETS" --dataset IMR90_HindIII --chroms $HUMAN_CHROMS --include histone
for res in 5000 50000 250000; do
  python -m probc preprocess --datasets-config "$DATASETS" --dataset IMR90 --chroms $HUMAN_CHROMS \
      --mode binned --resolution $res --include histone
done
python -m probc preprocess --datasets-config "$DATASETS" --dataset mESC --chroms $MOUSE_CHROMS

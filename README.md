# ProbC — joint modeling of epigenome and transcriptome effects in 3D genome

Reimplementation and replication package for

> Sefer E. **ProbC: joint modeling of epigenome and transcriptome effects in 3D genome.**
> *BMC Genomics* 23, 287 (2022). https://doi.org/10.1186/s12864-022-08498-5

ProbC decomposes Hi-C / Micro-C interactions into interactions between chromatin
marks (histone modifications, TF binding sites, DNase). Each pair of marks `(m, n)`
has an interaction probability `x_mn`; two restriction sites `u, v` interact with

```
P(E_uv | X) = 1 - Π_{m,n} (1 - x_mn)^(c^u_m c^v_n) = 1 - exp(-C_u^T Z C_v),   x_mn = 1 - exp(-z_mn)
```

and `Z ≥ 0` is estimated by convex penalised maximum likelihood over *both* interacting
and non-interacting pairs,

```
argmin_{Z ≥ 0}  -log L(Z, G) + λ_1 ||Z||_* + λ_2 ||Z||_1
```

solved with ADMM (singular-value thresholding for the nuclear norm, soft thresholding
for ℓ1, and L-BFGS-B or Armijo projected gradient descent for the smooth step). A
binned variant uses the Poisson (Le Cam) approximation `k_uv ~ Poisson(C_u^T X C_v)`.

---

## Contents

| | |
|---|---|
| [Install](#install) | [Quick start](#quick-start) |
| [Reproducing the paper](#reproducing-the-paper) | [Paper ↔ code map](#paper--code-map) |
| [Implementation notes](#implementation-notes-and-deviations) | [Synthetic validation](#synthetic-validation) |
| [Repository layout](#repository-layout) | [Citation](#citation) |

## Install

```bash
git clone https://github.com/seferlab/probc.git
cd probc
pip install -e .            # core: numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml
pip install -e ".[io]"      # + cooler (.cool/.mcool), hic-straw (.hic), pyBigWig
pip install -e ".[dev]"     # + pytest, ruff
```

Python ≥ 3.9. `hic-straw` needs a C++ compiler and libcurl headers.

## Quick start

```python
import numpy as np
from probc import ProbC, MarkMatrix, PairSet, Universe, build_full_pairset

# C: (sites x marks) peak counts near each restriction site; chrom/pos: site coordinates
mm = MarkMatrix(C, marks, chrom, pos)
edges = PairSet(u, v, np.ones(len(u)))            # observed interacting site pairs
universe = Universe("intra", min_dist=1, max_dist=2_000_000)

full = build_full_pairset(mm, edges, universe)    # all other pairs = non-interactions (exact, implicit)
model = ProbC(lambda_nuc=1e-3, lambda_l1=1e-3).fit(mm, full)

model.X_                     # mark-pair interaction probabilities x_mn
model.interaction_table()    # tidy table, strongest pairs first
model.predict_proba(mm, u_test, v_test)
```

Run the whole pipeline on synthetic stand-ins of the paper's datasets (no downloads;
about 8 minutes on 2 CPU cores, `--quick` takes about 2 minutes):

```bash
python examples/quickstart.py
python -m probc run all --synthetic --out-dir results/synthetic     # or: make synthetic
python -m probc run all --synthetic --quick --out-dir results/quick # smoke test
pytest                                                              # 36 tests
```

## Reproducing the paper

```bash
# 1. Download genomes (UCSC), peak files (ENCODE REST API, incl. Roadmap) and Hi-C/Micro-C
python -m probc download --datasets-config configs/datasets.yaml

# 2. Build restriction-site level datasets (peaks within 100 bp of a site; cached in data/processed)
bash scripts/preprocess_all.sh

# 3. Run every experiment (Figs. 1-13, Tables 2-3) -> results/paper/<experiment>/
python -m probc run all --config configs/experiments.yaml --out-dir results/paper
#    or one at a time:
python -m probc run prediction --config configs/experiments.yaml --out-dir results/paper
```

`scripts/run_paper.sh` chains the three steps.

**Before downloading**, open `configs/datasets.yaml`: the Hi-C sources are the GEO
series cited in the paper (GSE87112 hESC, GSE63525 IMR90 MboI, GSE35156 IMR90
HindIII, GSE130275 mESC Micro-C), but supplementary file names change over time,
so fill in `geo_file`/`url` (or point `path` to local copies). Entries marked
`VERIFY` need checking against the data actually used in the paper. Peak files are
located by ENCODE search (target + biosample + assembly, preferring the
`preferred_default` replicated / IDR-thresholded peaks); `peak_files:` overrides any
mark with an explicit URL (e.g. Roadmap `E003`/`E017` narrowPeak files), and every
download is recorded in `data/peaks/<dataset>/manifest.csv`.

Accepted Hi-C inputs: Juicer `merged_nodups.txt(.gz)` or 4DN `.pairs(.gz)` (read ends
assigned to the nearest restriction site — the paper's unbinned graph), Juicer `.hic`
(BP or FRAG resolution, via `hic-straw`) and `.cool`/`.mcool`. Micro-C has no
restriction sites; its nodes are fixed 200 bp bins.

Single fits:

```bash
python -m probc fit --dataset hESC --include histone --lambda-nuc 1e-3 --lambda-l1 1e-3 --output results/hESC_histone.npz
python -m probc fit --dataset hESC --support diagonal --output results/hESC_diag.npz   # forbid mark-mark interactions
python -m probc fit --dataset IMR90 --mode binned --resolution 50000 --likelihood poisson --include histone
```

## Paper ↔ code map

| Paper | Code | Output (`results/<run>/…`) |
|---|---|---|
| Eqs. 1–2, 4 (Bernoulli likelihood) | `probc/objectives.py::bernoulli_nll` | |
| Eq. 3, Theorem 1 (convex objective) | `probc/model.py::ProbC`, `tests/test_model.py::test_convexity_midpoint` | |
| Eqs. 5–17 (ADMM, SVT, soft-threshold, Armijo GD) | `probc/admm.py`, `probc/prox.py` | |
| Eqs. 18–19 (binned, Poisson / Le Cam) | `probc/objectives.py::poisson_nll`, `likelihood="poisson"` | |
| Datasets & preprocessing (100 bp mapping, 1 kb RPKM binning) | `probc/io/*`, `probc/pipeline.py`, `configs/datasets.yaml` | `data/processed/*.npz` |
| 5-fold nested CV (4-fold inner) | `probc/cv.py` | |
| Fig. 1, "forbidden interactions", TFBS vs histone coverage | `experiments/mark_selection.py` | `mark_selection/` |
| Figs. 2–3, Table 2, ES–IMR90 correlation | `experiments/heatmaps.py` | `heatmaps/` |
| Fig. 4 (chromosome-pair AUC), Fig. 5 (Venn), Fig. 6 (10 kb maps) | `experiments/prediction.py` | `prediction/` |
| Fig. 7 (vs HiC-Reg), Fig. 8 (vs distance decay) | `experiments/prediction.py`, `probc/baselines.py` | `prediction/` |
| Fig. 9 (TAD/compartment NVI), Table 3 (boundaries) | `experiments/domains.py`, `probc/tads.py`, `probc/compartments.py` | `domains/` |
| Perturbations (Sox9–Kcnj2 CTCF, FLT3 DS2 deletion), SHAP | `experiments/perturbation.py`, `probc/perturb.py`, `probc/attribution.py` | `perturbation/` |
| Figs. 10–11 (cross-species / cross-cell type) | `experiments/transfer.py` | `transfer/` |
| Fig. 12 (λ robustness) | `experiments/robustness.py` | `robustness/` |
| Fig. 13 (enzyme replicates, resolutions) | `experiments/stability.py` | `stability/` |

## Repository layout

```
probc/
  model.py          ProbC estimator (fit / predict / save / load)
  objectives.py     Bernoulli & Poisson NLLs + gradients, pattern compression
  admm.py, prox.py  ADMM solver and proximal operators
  data.py           MarkMatrix, PairSet, pair universes, negative sampling
  cv.py             nested CV, chromosome-pair / cross-dataset AUC matrices
  baselines.py      distance decay, HiC-Reg-style random forest
  tads.py           Armatus-style TAD calling (+ binary wrapper)
  compartments.py   A/B compartments
  metrics.py        AUC, NVI, boundary confusion, Spearman, SCC (HiCRep)
  perturb.py        mark masking, region deletion
  attribution.py    exact Shapley attributions
  binning.py        site -> bin aggregation, contact matrices
  synthetic.py      generative simulator
  io/               genome (restriction sites), peaks, Hi-C readers, downloads
  pipeline.py       raw files -> cached datasets
  experiments/      one module per Results section
  cli.py            `python -m probc {download,preprocess,fit,run,simulate}`
configs/            datasets.yaml (sources), experiments.yaml (all settings)
scripts/            preprocess_all.sh, run_paper.sh
examples/           quickstart.py
tests/              pytest suite (gradients, convexity, recovery, IO, metrics, experiments)
docs/               synthetic results
```

## Citation

```bibtex
@article{sefer2022probc,
  title   = {ProbC: joint modeling of epigenome and transcriptome effects in 3D genome},
  author  = {Sefer, Emre},
  journal = {BMC Genomics},
  volume  = {23},
  pages   = {287},
  year    = {2022},
  doi     = {10.1186/s12864-022-08498-5}
}
```

Funding acknowledged in the paper: Ozyegin University Research Grant; TÜBİTAK 3501
project 122E706. Code released under the MIT License.

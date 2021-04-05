"""End-to-end test of the real-data path on tiny fake raw files:
FASTA -> restriction sites, BED peaks -> 100 bp site mapping, .pairs -> contacts,
then dataset caching, the CLI `fit` command and one experiment in non-synthetic mode."""
import gzip

import numpy as np
import pandas as pd
import pytest
import yaml

from probc.data import MarkMatrix, PairSet
from probc.io.genome import restriction_sites_from_fasta
from probc.io.hic import write_pairs
from probc.pipeline import build_dataset
from probc.synthetic import planted_theta

MARKS = {"histone": ["H3K4me1", "H3K4me3", "H3K9me3", "H3K27ac"], "tf": ["CTCF", "RAD21"]}


def _fake_raw(root, name, seed):
    rng = np.random.default_rng(seed)
    (root / "genome").mkdir(parents=True, exist_ok=True)
    chroms = {"chr1": 300_000, "chr2": 250_000}
    fasta = root / "genome" / f"{name}.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        for c, L in chroms.items():
            fh.write(f">{c}\n")
            seq = "".join(rng.choice(list("ACGT"), size=L))
            for i in range(0, L, 80):
                fh.write(seq[i:i + 80] + "\n")
    sites = restriction_sites_from_fasta(fasta, "MboI")
    marks = MARKS["histone"] + MARKS["tf"]
    rows = {m: [] for m in marks}
    Cs, chs, ps = [], [], []
    for c, pos in sites.items():
        n = len(pos)
        dom = np.repeat(rng.integers(0, 3, size=n // 40 + 1), 40)[:n]
        E = rng.random((3, len(marks))) * 0.25
        C = (rng.random((n, len(marks))) < E[dom]).astype(float)
        for j, m in enumerate(marks):
            for p in pos[C[:, j] > 0]:
                rows[m].append((c, max(0, p - 40), p + 40))
        Cs.append(C)
        chs.append(np.full(n, c))
        ps.append(pos)
    mm = MarkMatrix(np.vstack(Cs), marks, np.concatenate(chs), np.concatenate(ps))
    (root / "peaks" / name).mkdir(parents=True, exist_ok=True)
    for m, r in rows.items():
        pd.DataFrame(r).to_csv(root / "peaks" / name / f"{m}.bed.gz", sep="\t", header=False, index=False)
    theta = planted_theta(marks, ["H3K27ac", "H3K4me3", "CTCF"], rng, scale=0.3)
    us, vs = [], []
    A = mm.C @ theta
    for c in chroms:
        idx = np.flatnonzero(mm.chrom == c)
        for k, i in enumerate(idx[:-1]):
            j = idx[k + 1: k + 120]
            p = -np.expm1(-(A[i] @ mm.C[j].T))
            hit = rng.random(len(j)) < p
            us.append(np.full(hit.sum(), i))
            vs.append(j[hit])
    edges = PairSet(np.concatenate(us), np.concatenate(vs), np.ones(sum(len(x) for x in us)))
    (root / "hic" / name).mkdir(parents=True, exist_ok=True)
    write_pairs(root / "hic" / name / "c.pairs", mm, edges)
    return {"species": "human", "assembly": "hg38", "encode_assembly": "GRCh38", "biosample": "X",
            "genome_fasta": f"genome/{name}.fa.gz", "chromosomes": list(chroms),
            "hic": {"format": "pairs", "path": f"hic/{name}/c.pairs", "enzyme": "MboI"},
            "histone": MARKS["histone"], "tf": MARKS["tf"], "dnase": False}, theta, len(edges)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    ds_cfg = {}
    ds_cfg["hESC"], theta, n_edges = _fake_raw(root, "hESC", 0)
    ds_cfg["IMR90"], _, _ = _fake_raw(root, "IMR90", 1)
    dcfg = {"data_root": str(root), "datasets": ds_cfg, "universe": {"scope": "intra", "min_dist": 1, "max_dist": 200_000}}
    with open(root / "datasets.yaml", "w") as fh:
        yaml.safe_dump(dcfg, fh)
    return root, dcfg, theta, n_edges


def test_build_dataset(raw):
    root, dcfg, theta, n_edges = raw
    ds = build_dataset(dcfg, "hESC", ["chr1", "chr2"])
    assert ds.marks.n_marks == 6 and len(ds.edges) > 0.95 * n_edges
    assert (root / "processed").exists()
    ds_h = build_dataset(dcfg, "hESC", ["chr1", "chr2"], include=("histone",))
    assert ds_h.marks.marks == MARKS["histone"]
    ds_b = build_dataset(dcfg, "hESC", ["chr1"], mode="binned", resolution=10_000)
    assert ds_b.marks.n_sites <= 31 and ds_b.edges.y.max() > 1


def test_cli_fit_and_experiment(raw, tmp_path):
    from probc import experiments
    from probc.cli import main
    from probc.experiments.common import make_context

    root, dcfg, theta, _ = raw
    cfg = {"datasets_config": str(root / "datasets.yaml"), "universe": dcfg["universe"],
           "grid": {"lambda_nuc": [1e-3], "lambda_l1": [1e-3]}, "cv": {"outer_folds": 3, "inner_folds": 2}}
    with open(tmp_path / "exp.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    main(["fit", "--config", str(tmp_path / "exp.yaml"), "--dataset", "hESC", "--output", str(tmp_path / "m.npz")])
    from probc import ProbC

    m = ProbC.load(tmp_path / "m.npz")
    assert set(m.active_marks(0.1 * m.theta_.max())) <= {"H3K27ac", "H3K4me3", "CTCF", "H3K4me1", "H3K9me3", "RAD21"}
    ctx = make_context(str(tmp_path / "exp.yaml"), synthetic=False, outdir=str(tmp_path / "hm"), quick=True)
    out = experiments.get("heatmaps").run(ctx)
    assert "pearson" in out

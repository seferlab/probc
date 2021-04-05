"""Dataset construction from raw files (configured in ``configs/datasets.yaml``).

A *dataset* is a pair (MarkMatrix, PairSet of observed interactions) at either the
restriction-site level (unbinned; the paper's default) or at a fixed bin size.
Processed datasets are cached as ``.npz`` files under ``<data_root>/processed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import yaml

from .binning import bin_ids, bin_pairs
from .data import MarkMatrix, PairSet, Universe, remap_pairs


@dataclass
class Dataset:
    name: str
    marks: MarkMatrix
    edges: PairSet
    universe: Universe
    meta: dict = field(default_factory=dict)

    @property
    def mark_names(self):
        return self.marks.marks


def load_config(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def save_dataset(ds: Dataset, path) -> None:
    mm = ds.marks
    np.savez_compressed(
        path, C=mm.C, marks=np.array(mm.marks), chrom=mm.chrom, pos=mm.pos,
        end=mm.end if mm.end is not None else np.zeros(0, np.int64),
        u=ds.edges.u, v=ds.edges.v, y=ds.edges.y,
        universe=np.array([ds.universe.scope, str(ds.universe.min_dist), str(ds.universe.max_dist)]),
        name=np.array(ds.name),
    )


def load_dataset(path) -> Dataset:
    d = np.load(path, allow_pickle=False)
    end = d["end"] if len(d["end"]) else None
    mm = MarkMatrix(d["C"], [str(m) for m in d["marks"]], d["chrom"], d["pos"], end)
    scope, mind, maxd = [str(x) for x in d["universe"]]
    uni = Universe(scope, int(mind), None if maxd == "None" else int(maxd))
    return Dataset(str(d["name"]), mm, PairSet(d["u"], d["v"], d["y"]), uni)


def dataset_marks(cfg_ds: dict, include: Sequence[str] = ("histone", "tf", "dnase")) -> Dict[str, str]:
    """mark name -> kind ("histone" | "tf" | "dnase")."""
    out = {}
    if "histone" in include:
        for m in cfg_ds.get("histone", []):
            out[m] = "histone"
    if "tf" in include:
        for m in cfg_ds.get("tf", []):
            out[m] = "tf"
    if "dnase" in include and cfg_ds.get("dnase", False):
        out["DNase"] = "dnase"
    return out


def peak_path(data_root: Path, ds_name: str, mark: str) -> Path:
    return Path(data_root) / "peaks" / ds_name / f"{mark}.bed.gz"


def _nodes(cfg_ds: dict, data_root: Path, chroms: Sequence[str]):
    """Restriction sites (or fixed Micro-C bins) per chromosome."""
    from .io.genome import fixed_bins, read_chrom_sizes, read_juicer_site_file, restriction_sites_from_fasta

    nodes = cfg_ds.get("nodes", {"type": "restriction_sites"})
    if nodes.get("type") == "fixed_bins":
        sizes = read_chrom_sizes(Path(data_root) / cfg_ds["chrom_sizes"])
        return fixed_bins({c: sizes[c] for c in chroms}, int(nodes["size"]))
    site_file = cfg_ds.get("restriction_sites")
    if site_file and (Path(data_root) / site_file).exists():
        return read_juicer_site_file(Path(data_root) / site_file, chroms)
    fasta = Path(data_root) / cfg_ds["genome_fasta"]
    return restriction_sites_from_fasta(fasta, cfg_ds["hic"]["enzyme"], chroms)


def build_dataset(
    cfg: dict,
    name: str,
    chroms: Sequence[str],
    mode: str = "unbinned",
    resolution: Optional[int] = None,
    include: Sequence[str] = ("histone", "tf", "dnase"),
    max_dist_to_site: int = 100,
    binary_marks: bool = False,
    min_count: float = 1.0,
    cache: bool = True,
) -> Dataset:
    """Build a dataset from raw files described in the config."""
    from .io.hic import contacts_from_cool, contacts_from_hic, contacts_from_pairs
    from .io.peaks import read_bed, site_mark_matrix

    data_root = Path(cfg.get("data_root", "data"))
    cfg_ds = cfg["datasets"][name]
    tag = f"{name}_{mode}{'' if resolution is None else '_' + str(resolution)}_{'-'.join(include)}_{'-'.join(chroms)}"
    cache_path = data_root / "processed" / f"{tag}.npz"
    if cache and cache_path.exists():
        return load_dataset(cache_path)

    marks = dataset_marks(cfg_ds, include)
    peaks = {}
    for m in marks:
        p = peak_path(data_root, name, m)
        if not p.exists():
            raise FileNotFoundError(f"Missing peaks for {name}/{m}: {p} (run scripts/download_data.py)")
        peaks[m] = read_bed(p, chroms)
    nodes = _nodes(cfg_ds, data_root, chroms)
    nodes = {c: nodes[c] for c in chroms if c in nodes}
    node_end = None
    if cfg_ds.get("nodes", {}).get("type") == "fixed_bins":
        size = int(cfg_ds["nodes"]["size"])
        node_end = {c: p + size - 1 for c, p in nodes.items()}
        max_dist_to_site = 0
    mm = site_mark_matrix(nodes, peaks, max_dist=max_dist_to_site, node_end=node_end)

    hic = cfg_ds["hic"]
    hic_path = data_root / hic["path"]
    fmt = hic["format"]
    if fmt in ("pairs", "merged_nodups"):
        edges = contacts_from_pairs(hic_path, mm, fmt=fmt, min_mapq=hic.get("min_mapq", 30))
    elif fmt == "hic":
        unit = "FRAG" if mode == "unbinned" else "BP"
        target = mm if unit == "FRAG" else bin_ids(mm, resolution)[1]
        edges = contacts_from_hic(hic_path, target, chroms, resolution or 1, unit=unit)
    elif fmt == "cool":
        uri = str(hic_path)
        if uri.endswith(".mcool"):
            uri = f"{uri}::resolutions/{resolution or cfg_ds['nodes']['size']}"
        target = mm if resolution is None else bin_ids(mm, resolution)[1]
        edges = contacts_from_cool(uri, target, chroms)
    else:
        raise ValueError(fmt)

    if mode == "binned":
        assert resolution, "binned mode needs a resolution"
        s2b, binned = bin_ids(mm, resolution)
        if fmt in ("pairs", "merged_nodups"):
            edges = bin_pairs(edges, s2b)
        mm = binned
    if binary_marks:
        mm = mm.binarize(0)
    n_before = mm.n_sites
    mm, kept = mm.drop_empty_sites()  # sites without marks cannot explain interactions
    edges = remap_pairs(edges, kept, n_before)
    edges = PairSet(edges.u, edges.v, edges.y)
    if mode == "unbinned":
        edges = edges.binary(min_count)
    uni_cfg = cfg.get("universe", {})
    universe = Universe(uni_cfg.get("scope", "intra"), int(uni_cfg.get("min_dist", 1)),
                        None if uni_cfg.get("max_dist") is None else int(uni_cfg["max_dist"]))
    ds = Dataset(name, mm, edges, universe, dict(mode=mode, resolution=resolution))
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(ds, cache_path)
    return ds

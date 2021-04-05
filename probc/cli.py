"""Command-line interface: ``probc {download,preprocess,fit,run,simulate}``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("probc")


# ----------------------------------------------------------------------------------
def cmd_download(args):
    from .io.download import (download, encode_download, encode_peak_file, geo_suppl_url, ucsc_chrom_sizes_url,
                              ucsc_genome_url)
    from .pipeline import dataset_marks, load_config, peak_path

    cfg = load_config(args.datasets_config)
    root = Path(cfg.get("data_root", "data"))
    names = args.datasets or list(cfg["datasets"])
    for name in names:
        d = cfg["datasets"][name]
        log.info("== %s ==", name)
        if "genome" in args.what:
            if d.get("genome_fasta"):
                download(ucsc_genome_url(d["assembly"]), root / d["genome_fasta"])
            if d.get("chrom_sizes"):
                download(ucsc_chrom_sizes_url(d["assembly"]), root / d["chrom_sizes"])
        if "peaks" in args.what:
            manifest = []
            for mark, kind in dataset_marks(d).items():
                dest = peak_path(root, name, mark)
                if dest.exists():
                    continue
                override = (d.get("peak_files") or {}).get(mark)
                if override:
                    download(override, dest)
                    manifest.append({"mark": mark, "source": override})
                    continue
                f = encode_peak_file(mark, d["biosample"], d["encode_assembly"], kind)
                if f is None:
                    log.warning("no ENCODE peaks for %s %s in %s", name, mark, d["biosample"])
                    continue
                tmp = encode_download(f, dest.parent / "_raw")
                tmp.rename(dest)
                manifest.append({"mark": mark, "accession": f.get("accession"), "output_type": f.get("output_type"),
                                 "dataset": f.get("dataset")})
                log.info("%s: %s (%s)", mark, f.get("accession"), f.get("output_type"))
            if manifest:
                import pandas as pd

                mp = root / "peaks" / name / "manifest.csv"
                old = pd.read_csv(mp) if mp.exists() else None
                df = pd.DataFrame(manifest) if old is None else pd.concat([old, pd.DataFrame(manifest)])
                df.to_csv(mp, index=False)
        if "hic" in args.what:
            h = d["hic"]
            url = h.get("url")
            if url is None and h.get("geo") and h.get("geo_file"):
                url = geo_suppl_url(h["geo"], h["geo_file"])
            if url:
                download(url, root / h["path"])
            else:
                log.warning("%s: no Hi-C URL configured; place the file at %s", name, root / h["path"])


def cmd_preprocess(args):
    from .pipeline import build_dataset, load_config

    cfg = load_config(args.datasets_config)
    ds = build_dataset(cfg, args.dataset, args.chroms, mode=args.mode, resolution=args.resolution,
                       include=args.include, max_dist_to_site=args.max_dist, cache=True)
    log.info("%s: %d nodes, %d marks, %d interacting pairs", ds.name, ds.marks.n_sites, ds.marks.n_marks, len(ds.edges))


def cmd_fit(args):
    from .data import build_full_pairset
    from .experiments.common import make_context

    ctx = make_context(args.config, args.synthetic, args.out_dir)
    ds = ctx.dataset(args.dataset, mode=args.mode, resolution=args.resolution, include=args.include)
    if args.chroms:
        from .experiments.common import chrom_dataset

        ds = chrom_dataset(ds, args.chroms)
    full = build_full_pairset(ds.marks, ds.edges, ds.universe)
    model = ctx.base_model(lambda_nuc=args.lambda_nuc, lambda_l1=args.lambda_l1, support=args.support,
                           likelihood=args.likelihood, verbose=args.verbose)
    t = time.time()
    model.fit(ds.marks, full)
    log.info("fitted in %.1fs (%d ADMM iterations, converged=%s)", time.time() - t, model.n_iter_, model.converged_)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    model.interaction_table(tol=1e-6).to_csv(out.with_suffix(".interactions.csv"), index=False)
    print(model.interaction_table(tol=1e-6).head(15).to_string(index=False))


def cmd_run(args):
    from . import experiments
    from .experiments.common import make_context

    names = experiments.EXPERIMENTS if args.experiment == "all" else [args.experiment]
    summary = {}
    for name in names:
        outdir = Path(args.out_dir) / name
        ctx = make_context(args.config, args.synthetic, str(outdir), quick=args.quick)
        t = time.time()
        log.info("=== %s ===", name)
        experiments.get(name).run(ctx)
        summary[name] = round(time.time() - t, 1)
        log.info("=== %s finished in %.1fs ===", name, summary[name])
    tp = Path(args.out_dir) / "timings.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    old.update(summary)
    with open(tp, "w") as fh:
        json.dump(old, fh, indent=2)


def cmd_simulate(args):
    from .pipeline import Dataset, save_dataset
    from .synthetic import simulate_genome

    g = simulate_genome(n_chroms=args.n_chroms, sites_per_chrom=args.sites, seed=args.seed,
                        likelihood=args.likelihood, distance_decay=args.distance_decay)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(Dataset("synthetic", g.marks, g.edges, g.universe), out)
    np.save(out.with_suffix(".theta_true.npy"), g.theta_true)
    log.info("wrote %s (%d sites, %d interactions)", out, g.marks.n_sites, len(g.edges))


# ----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="probc", description="ProbC (Sefer 2022) reimplementation")
    p.add_argument("-v", "--verbose", action="count", default=1)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download genomes, ENCODE peaks and Hi-C files")
    d.add_argument("--datasets-config", default="configs/datasets.yaml")
    d.add_argument("--datasets", nargs="*")
    d.add_argument("--what", nargs="*", default=["genome", "peaks", "hic"], choices=["genome", "peaks", "hic"])
    d.set_defaults(func=cmd_download)

    pp = sub.add_parser("preprocess", help="build and cache a processed dataset")
    pp.add_argument("--datasets-config", default="configs/datasets.yaml")
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--chroms", nargs="+", required=True)
    pp.add_argument("--mode", default="unbinned", choices=["unbinned", "binned"])
    pp.add_argument("--resolution", type=int)
    pp.add_argument("--include", nargs="+", default=["histone", "tf", "dnase"])
    pp.add_argument("--max-dist", type=int, default=100, help="max distance (bp) between a peak and a restriction site")
    pp.set_defaults(func=cmd_preprocess)

    f = sub.add_parser("fit", help="fit ProbC on one dataset with fixed lambdas")
    f.add_argument("--config", default="configs/experiments.yaml")
    f.add_argument("--synthetic", action="store_true")
    f.add_argument("--dataset", default="hESC")
    f.add_argument("--chroms", nargs="*")
    f.add_argument("--mode", default="unbinned", choices=["unbinned", "binned"])
    f.add_argument("--resolution", type=int)
    f.add_argument("--include", nargs="+", default=["histone", "tf", "dnase"])
    f.add_argument("--lambda-nuc", type=float, default=1e-3)
    f.add_argument("--lambda-l1", type=float, default=1e-3)
    f.add_argument("--support", default="full", choices=["full", "diagonal"])
    f.add_argument("--likelihood", default="bernoulli", choices=["bernoulli", "poisson"])
    f.add_argument("--output", default="results/model.npz")
    f.add_argument("--out-dir", default="results")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("run", help="run an experiment (or 'all')")
    r.add_argument("experiment")
    r.add_argument("--config", default="configs/experiments.yaml")
    r.add_argument("--synthetic", action="store_true", help="use synthetic stand-ins for the paper's datasets")
    r.add_argument("--quick", action="store_true", help="smaller grids / fewer chromosomes (smoke test)")
    r.add_argument("--out-dir", default="results")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="write a synthetic dataset")
    s.add_argument("--output", default="data/synthetic/synthetic.npz")
    s.add_argument("--n-chroms", type=int, default=3)
    s.add_argument("--sites", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--likelihood", default="bernoulli", choices=["bernoulli", "poisson"])
    s.add_argument("--distance-decay", type=float)
    s.set_defaults(func=cmd_simulate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "fit":
        args.verbose = 0 if args.verbose <= 1 else 10
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

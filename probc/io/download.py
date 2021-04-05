"""Data download helpers (ENCODE REST API, GEO supplementary files, UCSC).

ENCODE peak files are located with the public search API instead of hard-coded
accessions, so the same configuration works for any target / biosample that ENCODE
(including the Roadmap reference epigenomes it now hosts) provides.
"""
from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

ENCODE = "https://www.encodeproject.org"
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"

HISTONE_OUTPUT_TYPES = ("replicated peaks", "pseudoreplicated peaks", "stable peaks", "peaks")
TF_OUTPUT_TYPES = ("conservative IDR thresholded peaks", "optimal IDR thresholded peaks",
                   "IDR thresholded peaks", "pseudoreplicated peaks", "peaks")
DNASE_OUTPUT_TYPES = ("peaks", "replicated peaks", "pseudoreplicated peaks")


def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def encode_search_files(
    biosample: str,
    assembly: str,
    target: Optional[str] = None,
    assay_title: Optional[str] = None,
    file_format: str = "bed",
    extra: Optional[dict] = None,
) -> list:
    params = [
        ("type", "File"), ("status", "released"), ("file_format", file_format), ("assembly", assembly),
        ("biosample_ontology.term_name", biosample), ("format", "json"), ("limit", "all"),
    ]
    if target:
        params.append(("target.label", target))
    if assay_title:
        params.append(("assay_title", assay_title))
    for k, v in (extra or {}).items():
        params.append((k, v))
    url = f"{ENCODE}/search/?" + urllib.parse.urlencode(params)
    try:
        return _get_json(url).get("@graph", [])
    except urllib.error.HTTPError as e:  # ENCODE returns 404 for empty searches
        if e.code == 404:
            return []
        raise


def pick_peak_file(files: list, output_types: Sequence[str]) -> Optional[dict]:
    """Prefer ENCODE's 'preferred_default' file, then the output-type order given."""
    if not files:
        return None
    pref = [f for f in files if f.get("preferred_default")]
    for ot in output_types:
        cands = [f for f in (pref or files) if f.get("output_type") == ot]
        if cands:
            # most replicates first, then most recent
            cands.sort(key=lambda f: (len(f.get("biological_replicates", [])), f.get("date_created", "")), reverse=True)
            return cands[0]
    return files[0]


def encode_peak_file(mark: str, biosample: str, assembly: str, kind: str) -> Optional[dict]:
    """Find the best peak file for a histone mark / TF / DNase in a biosample."""
    if kind == "histone":
        files = encode_search_files(biosample, assembly, target=mark, assay_title="Histone ChIP-seq")
        return pick_peak_file(files, HISTONE_OUTPUT_TYPES)
    if kind == "tf":
        files = encode_search_files(biosample, assembly, target=mark, assay_title="TF ChIP-seq")
        return pick_peak_file(files, TF_OUTPUT_TYPES)
    if kind == "dnase":
        files = encode_search_files(biosample, assembly, assay_title="DNase-seq")
        return pick_peak_file(files, DNASE_OUTPUT_TYPES)
    raise ValueError(kind)


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.rename(dest)
    return dest


def encode_download(file_obj: dict, outdir: Path) -> Path:
    href = file_obj["href"]
    name = Path(href).name
    return download(ENCODE + href, Path(outdir) / name)


def geo_suppl_url(gse: str, filename: str) -> str:
    """URL of a GEO series supplementary file."""
    num = gse[3:]
    stub = "GSE" + (num[:-3] if len(num) > 3 else "") + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/suppl/{filename}"


def ucsc_genome_url(assembly: str) -> str:
    return f"{UCSC}/{assembly}/bigZips/{assembly}.fa.gz"


def ucsc_chrom_sizes_url(assembly: str) -> str:
    return f"{UCSC}/{assembly}/bigZips/{assembly}.chrom.sizes"

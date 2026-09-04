#!/usr/bin/env python3
"""Download and merge PrimeKG + Hetionet into KnowGuard-compatible triplet CSV.

KnowGuard CSV schema (matches know_storage / proxy KG):
  relevant_description, x_name, y_name, relationship, image_path
where historically:
  x_name = head, y_name = relation, relationship = tail
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_HF_REPO = "hieupth/primekg"
PRIMEKG_HF_FILE = "org/kg.csv"
HETIONET_EDGES_URL = (
    "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
)
HETIONET_NODES_URL = (
    "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-nodes.tsv"
)

# Prefer clinically useful edges for MedQA-style reasoning.
PRIMEKG_KEEP_RELATIONS = {
    "indication",
    "contraindication",
    "off-label use",
    "drug_drug",
    "drug_protein",
    "disease_protein",
    "disease_disease",
    "phenotype_protein",
    "phenotype_phenotype",
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "exposure_disease",
    "exposure_disease",
    "anatomy_protein_present",
    "anatomy_protein_absent",
    "pathway_protein",
    "bioprocess_protein",
    "molfunc_protein",
    "cellcomp_protein",
    "exposure_protein",
    "exposure_bioprocess",
    "exposure_molfunc",
    "exposure_cellcomp",
}

HETIONET_KEEP_KINDS = {
    "treats",
    "palliates",
    "presents",
    "associates",
    "localizes",
    "causes",
    "resembles",
    "upregulates",
    "downregulates",
    "binds",
    "includes",
    "covaries",
    "interacts",
    "participates",
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 KnowGuardKG/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def ensure_primekg(dest: Path) -> Path:
    """Prefer local file; else Dataverse; else HuggingFace mirror."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    try:
        return download(PRIMEKG_URL, dest)
    except Exception as e:
        print(f"Dataverse download failed ({e}); trying HuggingFace {PRIMEKG_HF_REPO}")
    from huggingface_hub import hf_hub_download
    import shutil

    path = hf_hub_download(
        repo_id=PRIMEKG_HF_REPO, repo_type="dataset", filename=PRIMEKG_HF_FILE
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    print(f"Copied HF PrimeKG to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def row_triplet(head: str, relation: str, tail: str, source: str) -> dict:
    head = " ".join(str(head).split())[:200]
    relation = " ".join(str(relation).split())[:120]
    tail = " ".join(str(tail).split())[:200]
    desc = f"{head} --{relation}--> {tail} [{source}]"
    return {
        "relevant_description": desc,
        "x_name": head,
        "y_name": relation,
        "relationship": tail,
        "image_path": "",
        "source": source,
    }


def load_primekg(path: Path, max_edges: int | None) -> list[dict]:
    print(f"Loading PrimeKG from {path}")
    usecols = None
    # Peek columns
    peek = pd.read_csv(path, nrows=2)
    cols = {c.lower(): c for c in peek.columns}
    x_col = cols.get("x_name")
    y_col = cols.get("y_name")
    rel_col = cols.get("display_relation") or cols.get("relation")
    xt = cols.get("x_type")
    yt = cols.get("y_type")
    if not (x_col and y_col and rel_col):
        raise RuntimeError(f"Unexpected PrimeKG columns: {list(peek.columns)}")
    usecols = [x_col, y_col, rel_col] + ([xt] if xt else []) + ([yt] if yt else [])

    clinical = {"disease", "drug", "effect/phenotype", "anatomy", "exposure"}
    rows: list[dict] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
        for _, r in chunk.iterrows():
            relation = str(r[rel_col]).strip().lower()
            keep = relation in PRIMEKG_KEEP_RELATIONS
            if not keep and xt and yt:
                types = {str(r[xt]).lower(), str(r[yt]).lower()}
                keep = bool(types & clinical)
            if not keep:
                continue
            rows.append(row_triplet(r[x_col], relation, r[y_col], source="primekg"))
            if max_edges and len(rows) >= max_edges:
                print(f"PrimeKG kept {len(rows)} edges (capped)")
                return rows
    print(f"PrimeKG kept {len(rows)} edges")
    return rows


def load_hetionet(edges_path: Path, nodes_path: Path, max_edges: int | None) -> list[dict]:
    print(f"Loading Hetionet nodes from {nodes_path}")
    nodes = pd.read_csv(nodes_path, sep="\t")
    # id, name, kind
    id_to_name = {}
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        name = str(r.get("name", nid))
        kind = str(r.get("kind", ""))
        id_to_name[nid] = f"{name} ({kind})" if kind else name

    print(f"Loading Hetionet edges from {edges_path}")
    open_fn = gzip.open if str(edges_path).endswith(".gz") else open
    rows: list[dict] = []
    with open_fn(edges_path, "rt", encoding="utf-8") as f:
        # SIF-like: source\tmetaedge\ttarget  OR source metaedge target
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            if len(parts) < 3:
                continue
            src, kind, tgt = parts[0], parts[1], parts[2]
            kind_l = kind.lower().replace(">", "").strip()
            # metaedge often like CtD / treats — keep known verbs or disease-ish prefixes
            keep = any(k in kind_l for k in HETIONET_KEEP_KINDS) or kind_l in {
                "ctd",
                "cdp",
                "cpd",
                "daa",
                "dcu",
                "ddi",
                "dgu",
                "dil",
                "dpw",
                "dru",
                "dse",
                "gpc",
                "gpbp",
                "gpcc",
                "gpmf",
                "gr>g",
                "adg",
                "aec",
                "auc",
                "cbc",
                "ccse",
                "cdg",
                "cig",
                "crg",
                "cse",
                "ctd",
            }
            if not keep:
                # keep abbreviated metaedges that include Disease/Compound/Symptom letters
                keep = any(ch in kind for ch in ("D", "C", "S", "G", "A"))
            if not keep:
                continue
            head = id_to_name.get(src, src)
            tail = id_to_name.get(tgt, tgt)
            rows.append(row_triplet(head, kind, tail, source="hetionet"))
            if max_edges and len(rows) >= max_edges:
                break
    print(f"Hetionet kept {len(rows)} edges")
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (
            r["x_name"].lower(),
            r["y_name"].lower(),
            r["relationship"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_disease2demo(rows: list[dict], limit: int = 200) -> pd.DataFrame:
    diseases = []
    for r in rows:
        for field in ("x_name", "relationship"):
            val = str(r[field])
            low = val.lower()
            if "disease" in low or "(disease)" in low:
                diseases.append(val.split("(")[0].strip())
    diseases = sorted(set(diseases))[:limit]
    demographics = ["Adults", "Elderly", "Pregnant woman", "people with HIV"]
    out = []
    for d in diseases:
        for demo in demographics[:2]:
            out.append(
                {
                    "demographic": demo,
                    "disease": d,
                    "pdf_name": "{combined_guideline.pdf}",
                }
            )
    if not out:
        out = [
            {
                "demographic": "Adults",
                "disease": "general medical condition",
                "pdf_name": "{combined_guideline.pdf}",
            }
        ]
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/kg/raw")
    parser.add_argument("--output_dir", default="data/kg")
    parser.add_argument(
        "--max_edges",
        type=int,
        default=400000,
        help="Max total edges after merge (0 = unlimited)",
    )
    parser.add_argument(
        "--per_source_cap",
        type=int,
        default=250000,
        help="Max edges to keep while reading each source (0 = unlimited)",
    )
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_dataset").mkdir(parents=True, exist_ok=True)
    (out_dir / "WHO").mkdir(parents=True, exist_ok=True)

    prime_path = ensure_primekg(raw / "primekg_kg.csv")
    het_edges = download(HETIONET_EDGES_URL, raw / "hetionet-v1.0-edges.sif.gz")
    het_nodes = download(HETIONET_NODES_URL, raw / "hetionet-v1.0-nodes.tsv")

    per_cap = args.per_source_cap if args.per_source_cap > 0 else None
    rows = []
    rows.extend(load_primekg(prime_path, per_cap))
    rows.extend(load_hetionet(het_edges, het_nodes, per_cap))
    rows = dedupe(rows)

    if args.max_edges > 0 and len(rows) > args.max_edges:
        # Prefer primekg then hetionet already interleaved by append; keep head of list
        # but shuffle by source balance: take half from each if possible
        prime = [r for r in rows if r["source"] == "primekg"]
        het = [r for r in rows if r["source"] == "hetionet"]
        half = args.max_edges // 2
        rows = prime[:half] + het[: args.max_edges - half]
        rows = dedupe(rows)

    # Drop helper column before writing KnowGuard CSV
    df = pd.DataFrame(rows)
    sources = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    df_out = df.drop(columns=["source"], errors="ignore")
    csv_path = out_dir / "combined_primekg_hetionet.csv"
    df_out.to_csv(csv_path, index=False)

    demo_df = build_disease2demo(rows)
    demo_path = out_dir / "baseline_dataset" / "Disease2demo_combined.csv"
    demo_df.to_csv(demo_path, index=False)

    overview = [
        {
            "name": "combined_guideline.pdf",
            "png_dir_path": "WHO/combined/",
            "sources": ["primekg", "hetionet"],
        }
    ]
    with open(out_dir / "WHO" / "overview_combined.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    meta = {
        "n_triplets": len(df_out),
        "sources": sources,
        "csv": str(csv_path),
        "disease2demo": str(demo_path),
    }
    with open(out_dir / "combined_kg_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"Wrote combined KG to {csv_path}")


if __name__ == "__main__":
    main()

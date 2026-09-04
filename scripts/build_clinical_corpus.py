#!/usr/bin/env python3
"""Build MedQA context passage corpus + FAISS for clinical RAG."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def split_passages(text: str, max_chars: int = 600) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for s in sents:
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/kg/clinical_corpus")
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_chars", type=int, default=600)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    # Prefer train+validation contexts for richer textbook-like corpus
    passages: list[dict] = []
    seen = set()
    for split in ("train", "validation"):
        try:
            ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split=split)
        except Exception as e:
            print(f"Skip split {split}: {e}")
            continue
        for i, row in enumerate(ds):
            if len(passages) >= args.max_rows:
                break
            ctx = row.get("sent1") or row.get("context") or ""
            for j, chunk in enumerate(split_passages(ctx, max_chars=args.max_chars)):
                key = chunk.lower()
                if key in seen or len(chunk) < 40:
                    continue
                seen.add(key)
                passages.append(
                    {
                        "id": f"{split}-{i}-{j}",
                        "text": chunk,
                        "source": "medqa",
                        "split": split,
                    }
                )
            if len(passages) >= args.max_rows:
                break

    passages_path = out / "passages.jsonl"
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(passages)} passages -> {passages_path}")

    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(args.embedding_model)
    texts = [p["text"] for p in passages]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    np.save(out / "embeddings.npy", emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out / "index.faiss"))

    meta = {
        "n_passages": len(passages),
        "embedding_model": args.embedding_model,
        "dim": int(emb.shape[1]),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"FAISS ready: {out / 'index.faiss'} ({meta})")


if __name__ == "__main__":
    main()

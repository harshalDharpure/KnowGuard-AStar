#!/usr/bin/env python3
"""Clinical passage RAG: MedQA contexts + hybrid dense/BM25 + cross-encoder rerank."""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

_CORPUS_CACHE: dict[str, Any] = {}
_RAG_SINGLETON: ClinicalRAG | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class ClinicalRAG:
    """Retrieve textbook-style MedQA passages for KnowGuard text_knowledge."""

    def __init__(
        self,
        corpus_dir: str = "data/kg/clinical_corpus",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        use_rerank: bool = True,
        device: str | None = None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.device = device
        self.passages: list[str] = []
        self.meta: list[dict] = []
        self._embedder = None
        self._reranker = None
        self._index = None
        self._bm25 = None
        self._loaded = False

    def available(self) -> bool:
        return (self.corpus_dir / "passages.jsonl").exists() and (
            (self.corpus_dir / "index.faiss").exists()
            or (self.corpus_dir / "embeddings.npy").exists()
        )

    def load(self) -> None:
        if self._loaded:
            return
        key = str(self.corpus_dir.resolve())
        if key in _CORPUS_CACHE:
            cached = _CORPUS_CACHE[key]
            self.passages = cached["passages"]
            self.meta = cached["meta"]
            self._embedder = cached["embedder"]
            self._index = cached.get("index")
            self._embeddings = cached.get("embeddings")
            self._bm25 = cached.get("bm25")
            self._reranker = cached.get("reranker")
            self._loaded = True
            return

        passages_path = self.corpus_dir / "passages.jsonl"
        if not passages_path.exists():
            raise FileNotFoundError(f"Missing clinical corpus at {passages_path}")

        self.passages, self.meta = [], []
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.passages.append(row["text"])
                self.meta.append({k: v for k, v in row.items() if k != "text"})

        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)

        faiss_path = self.corpus_dir / "index.faiss"
        emb_path = self.corpus_dir / "embeddings.npy"
        if faiss_path.exists():
            import faiss

            self._index = faiss.read_index(str(faiss_path))
            self._embeddings = None
        elif emb_path.exists():
            self._embeddings = np.load(emb_path).astype("float32")
            self._index = None
        else:
            raise FileNotFoundError("Need index.faiss or embeddings.npy in clinical corpus")

        try:
            from rank_bm25 import BM25Okapi

            tokenized = [_tokenize(p) for p in self.passages]
            self._bm25 = BM25Okapi(tokenized)
        except Exception:
            self._bm25 = None

        if self.use_rerank:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model_name, device=self.device)
            except Exception:
                self._reranker = None

        _CORPUS_CACHE[key] = {
            "passages": self.passages,
            "meta": self.meta,
            "embedder": self._embedder,
            "index": self._index,
            "embeddings": getattr(self, "_embeddings", None),
            "bm25": self._bm25,
            "reranker": self._reranker,
        }
        self._loaded = True

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        if self._index is not None:
            scores, idxs = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        sims = (self._embeddings @ q[0]).tolist()
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])[:k]
        return [(i, float(s)) for i, s in ranked]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [(i, float(s)) for i, s in ranked]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        self.load()
        k = top_k or self.top_k
        cand_k = max(k * 4, 20)
        dense = self._dense_search(query, cand_k)
        sparse = self._bm25_search(query, cand_k)

        # Reciprocal rank fusion
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, (idx, _) in enumerate(sparse):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        fused = sorted(rrf.items(), key=lambda x: -x[1])[: max(cand_k, k * 2)]

        if self._reranker is not None and fused:
            pairs = [(query, self.passages[i]) for i, _ in fused[: cand_k]]
            scores = self._reranker.predict(pairs)
            order = sorted(range(len(fused[: cand_k])), key=lambda j: -float(scores[j]))
            fused = [(fused[j][0], float(scores[j])) for j in order]

        out = []
        for idx, score in fused[:k]:
            out.append(
                {
                    "text": self.passages[idx],
                    "score": score,
                    "meta": self.meta[idx] if idx < len(self.meta) else {},
                }
            )
        return out

    def format_knowledge(self, query: str, top_k: int | None = None) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        blocks = []
        for i, h in enumerate(hits, 1):
            blocks.append(f"[Clinical passage {i} | score={h['score']:.3f}]\n{h['text']}")
        return "\n\n".join(blocks)


def build_query_from_patient_state(patient_state: dict, inquiry: str = "") -> str:
    parts = [patient_state.get("initial_info", ""), inquiry or ""]
    for qa in patient_state.get("interaction_history", []):
        parts.append(qa.get("question", ""))
        parts.append(qa.get("answer", ""))
    return "\n".join(p for p in parts if p).strip()


def get_clinical_rag(args) -> ClinicalRAG | None:
    global _RAG_SINGLETON
    corpus_dir = getattr(args, "clinical_corpus_dir", "data/kg/clinical_corpus")
    if not getattr(args, "use_clinical_rag", False):
        return None
    if _RAG_SINGLETON is not None:
        return _RAG_SINGLETON
    rag = ClinicalRAG(
        corpus_dir=corpus_dir,
        embedding_model=getattr(
            args, "clinical_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_model=getattr(
            args, "clinical_reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        top_k=getattr(args, "clinical_rag_top_k", 5),
        use_rerank=getattr(args, "clinical_rag_rerank", True),
    )
    if not rag.available():
        return None
    rag.load()
    _RAG_SINGLETON = rag
    return rag

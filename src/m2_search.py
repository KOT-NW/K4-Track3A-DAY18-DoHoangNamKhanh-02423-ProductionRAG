from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic bag-of-words hashing embedding — fallback khi thiếu model."""
    import hashlib
    import math

    vec = [0.0] * dim
    for tok in segment_vietnamese(text).lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb + 1e-9)


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép"); BM25 tokenize bằng
    split(" ") nên phải replace("_", " ") để khớp với query.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c.get("text", "")).lower().split() for c in chunks
        ]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        """BM25Okapi-equivalent scorer dùng khi thiếu thư viện rank_bm25."""
        import math

        n = len(self.corpus_tokens)
        if n == 0:
            return []
        avgdl = sum(len(d) for d in self.corpus_tokens) / n
        df: dict[str, int] = {}
        for doc in self.corpus_tokens:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1

        k1, b = 1.5, 0.75
        scores = [0.0] * n
        for i, doc in enumerate(self.corpus_tokens):
            dl = len(doc) or 1
            for tok in query_tokens:
                tf = doc.count(tok)
                if tf == 0 or tok not in df:
                    continue
                idf = math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / (avgdl or 1)))
        return scores

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.corpus_tokens:
            return []
        query_tokens = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(query_tokens) if self.bm25 is not None else self._fallback_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in order if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        self._encoder = None
        self.client = None
        self._use_qdrant = False
        self._mem: dict = {}
        try:
            from qdrant_client import QdrantClient
            try:
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
                client.get_collections()
                self.client = client
                self._use_qdrant = True
            except Exception:
                self.client = QdrantClient(":memory:")
                self._use_qdrant = True
        except Exception:
            self.client = None
            self._use_qdrant = False
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._get_encoder().encode(texts, show_progress_bar=len(texts) > 1)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception:
            return [_hash_embed(t) for t in texts]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant (fallback: in-memory vectors)."""
        texts = [c.get("text", "") for c in chunks]
        vectors = self._encode(texts)
        self._mem[collection] = {"chunks": chunks, "vectors": vectors}

        if self._use_qdrant and self.client is not None:
            try:
                from qdrant_client.models import Distance, VectorParams, PointStruct
                self.client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                points = [
                    PointStruct(id=i, vector=v, payload={**c.get("metadata", {}), "text": c.get("text", "")})
                    for i, (c, v) in enumerate(zip(chunks, vectors))
                ]
                self.client.upsert(collection_name=collection, points=points)
            except Exception as e:
                print(f"  ⚠️  Qdrant indexing failed ({e}); using in-memory fallback.")
                self._use_qdrant = False

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._encode([query])[0]

        if self._use_qdrant and self.client is not None:
            try:
                response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata=pt.payload,
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as e:
                print(f"  ⚠️  Qdrant search failed ({e}); using in-memory fallback.")
                self._use_qdrant = False

        store = self._mem.get(collection)
        if not store:
            return []
        scored = sorted(
            ((_cosine(query_vector, vec), chunk) for chunk, vec in zip(store["chunks"], store["vectors"])),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            SearchResult(text=c.get("text", ""), score=float(s), metadata=c.get("metadata", {}), method="dense")
            for s, c in scored[:top_k]
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ordered = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in ordered
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

import numpy as np
import faiss


class Retriever:
    """Index a corpus of embeddings, then return top-k doc_ids per query."""

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> list[list[str]]: ...


class NaiveNumpyRetriever(Retriever):
    """NumPy brute-force baseline.

    For each query, scores the entire corpus via a single matmul and sorts all
    scores. Assumes embeddings are L2-normalized so dot product == cosine
    similarity. This is the reference against which approximate indexes
    (faiss IVF/HNSW/PQ) are compared.
    """

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.ids = list(ids)

    def search(self, queries: np.ndarray, k: int) -> list[list[str]]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores = q @ self.emb.T                       # (Q, N) cosine similarity
        top = np.argsort(-scores, axis=1)[:, :k]      # full sort per query
        return [[self.ids[i] for i in row] for row in top]


class FaissHNSWRetriever(Retriever):
    """FAISS HNSW (Hierarchical Navigable Small World) approximate index.

    Uses inner product as the metric, which equals cosine similarity for
    L2-normalized embeddings.

    Knobs:
      M:               neighbors per node in the graph (typical 16-64).
                       Higher = better recall, more memory, slower build.
      efConstruction:  candidate list size during build (typical 100-400).
                       Higher = better graph quality, slower build.
      efSearch:        candidate list size during search (typical 32-256).
                       Higher = better recall, slower search.
    """

    def __init__(self, M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        d = emb.shape[1]
        index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(emb)
        self.index = index
        self.ids = list(ids)

    def search(self, queries: np.ndarray, k: int) -> list[list[str]]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        _scores, idx = self.index.search(q, k)        # (Q, k) corpus positions
        return [[self.ids[i] for i in row if i != -1] for row in idx]


class HybridRRFRetriever:
    """Hybrid retrieval: BM25 (lexical) + dense (semantic) fused with RRF.

    All three pieces — BM25 index, dense matrix, and RRF fusion — live inside
    this single class. Build with both texts and embeddings; search with both
    query texts and query embeddings; one ranked list of doc_ids comes back.

    RRF score per doc = Σ 1 / (k_rrf + rank_i), summed over the two retrievers.
    k_rrf=60 is the value from Cormack et al. 2009. candidate_k controls how
    deep each side goes before fusion — bigger = better recall, slower.
    """

    def __init__(self, k_rrf: int = 60, candidate_k: int = 100):
        self.k_rrf = k_rrf
        self.candidate_k = candidate_k

    def build(self, texts: list[str], embeddings: np.ndarray, ids: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.ids = list(ids)

    def _bm25_topk(self, query_text: str, k: int) -> np.ndarray:
        scores = self.bm25.get_scores(query_text.lower().split())
        k = min(k, len(scores))
        top = np.argpartition(-scores, kth=k - 1)[:k]
        return top[np.argsort(-scores[top])]

    def _dense_topk(self, query_emb: np.ndarray, k: int) -> np.ndarray:
        q = np.ascontiguousarray(query_emb, dtype=np.float32).reshape(1, -1)
        scores = (q @ self.emb.T)[0]
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, kth=k - 1)[:k]
        return top[np.argsort(-scores[top])]

    def search(
        self,
        query_texts: list[str],
        query_embeddings: np.ndarray,
        k: int,
    ) -> list[list[str]]:
        c = self.candidate_k
        out = []
        for qt, qe in zip(query_texts, query_embeddings):
            bm_idx = self._bm25_topk(qt, c)
            dn_idx = self._dense_topk(qe, c)
            scores: dict[int, float] = {}
            for rank, i in enumerate(bm_idx, start=1):
                scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (self.k_rrf + rank)
            for rank, i in enumerate(dn_idx, start=1):
                scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (self.k_rrf + rank)
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
            out.append([self.ids[i] for i, _ in top])
        return out

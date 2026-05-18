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

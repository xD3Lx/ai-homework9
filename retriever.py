import numpy as np


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

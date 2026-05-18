import numpy as np


class Retriever:
    """Index a corpus of embeddings, then return top-k doc_ids per query."""

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> list[list[str]]: ...


class NaiveNumpyRetriever(Retriever):
    """Brute-force cosine similarity via a single (Q, N) matmul.

    Assumes embeddings are L2-normalized → dot product == cosine similarity.
    """

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.ids = list(ids)

    def search(self, queries: np.ndarray, k: int) -> list[list[str]]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, self.emb.shape[0])
        scores = q @ self.emb.T                                  # (Q, N)
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]  # unordered top-k
        rows = np.arange(scores.shape[0])[:, None]
        order = np.argsort(-scores[rows, part], axis=1)
        top = part[rows, order]                                  # (Q, k) ranked
        return [[self.ids[i] for i in row] for row in top]

import os
import time
import json
import psutil
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from template.data_loader import load_cache, build_subset
from template.metrics import evaluate

def check_process_ram():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    mb = 1024 ** 2
    return round(mem.rss / mb, 2)


def embed(
    texts: list[str],
    model: str | SentenceTransformer,
    batch_size: int = 64,
    device: str = "cpu",
    instruction: str = "",
) -> np.ndarray:
    """
    Encode a list of texts into a (N, D) float32 array of L2-normalized vectors.
    Normalized embeddings let you use faiss.IndexFlatIP as cosine similarity.

    Args:
        texts: input strings to encode.
        model: either a model name (e.g. "BAAI/bge-small-en-v1.5") or a preloaded
            SentenceTransformer instance.
        batch_size: encoding batch size.
        device: "cpu", "mps", or "cuda". If a model instance is passed it is
            moved to this device.
        instruction: optional prefix prepended to every text before encoding —
            e.g. BGE retrieval queries use
            "Represent this sentence for searching relevant passages: ".
            Pass "" for corpus passages.

    Returns:
        np.ndarray of shape (len(texts), D), dtype float32, L2-normalized.
    """
    if isinstance(model, str):
        model = SentenceTransformer(model, device=device)
    else:
        model = model.to(device)

    if instruction:
        texts = [instruction + t for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_cached(texts, cache_path, **kwargs):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb = embed(texts, **kwargs)
    np.save(cache_path, emb)
    return emb


def main():
    CACHE_PATH = Path("template/cache/corpus.json")

    pool, eval_set = load_cache(CACHE_PATH)

    print("Loading embedding model...")
    # Use MPS on Apple Silicon
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    # Cap sequence length — attention memory grows with seq^2.
    # MS MARCO passages fit comfortably in 256 tokens.
    model.max_seq_length = 256

    # BGE recommends an instruction prefix on queries but NOT on passages.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    # batch_size=512 OOMs on MPS; 64 is safe for bge-small at seq_len=256.
    BATCH_SIZE = 64

    print(f"Embedding {len(pool)} corpus docs...")
    t0 = time.time()
    corpus_embeddings = embed_cached(
        [d["text"] for d in pool],
        cache_path="template/cache/corpus_emb.npy",
        model=model,
        batch_size=BATCH_SIZE,
        device=device,
        instruction="",
    )
    print(f"  corpus_embeddings: {corpus_embeddings.shape}, {corpus_embeddings.dtype} "
          f"in {time.time() - t0:.1f}s, RAM={check_process_ram()} MB")

    # Release MPS buffers held by the corpus pass before embedding queries.
    if device == "mps":
        torch.mps.empty_cache()

    print(f"Embedding {len(eval_set)} eval queries...")
    t0 = time.time()
    query_embeddings = embed_cached(
        [e["query"] for e in eval_set],
        cache_path="template/cache/queries_emb.npy",
        model=model,
        batch_size=BATCH_SIZE,
        device=device,
        instruction=QUERY_INSTRUCTION,
    )
    print(f"  query_embeddings:  {query_embeddings.shape}, {query_embeddings.dtype} "
          f"in {time.time() - t0:.1f}s")

    sizes = [1000, 10000, 100000, 300000]
    results = {}


if __name__ == "__main__":
    main()
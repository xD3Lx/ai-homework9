import os
import csv
import time
import json
import psutil
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from template.data_loader import load_cache, build_subset
from template.metrics import evaluate
from retriever import HybridRRFRetriever

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

    id_to_pos = {d["id"]: i for i, d in enumerate(pool)}
    sizes = [1000, 10000, 100000, 300000]
    results = {}
    query_texts = [e["query"] for e in eval_set]

    for size in sizes:
        ram_before = check_process_ram()
        if size > len(pool):
            continue
        subset = build_subset(pool, eval_set, size)
        positions = [id_to_pos[d["id"]] for d in subset]
        subset_emb = corpus_embeddings[positions]
        subset_ids = [d["id"] for d in subset]
        subset_texts = [d["text"] for d in subset]

        retriever = HybridRRFRetriever(k_rrf=60, candidate_k=100)
        retriever.build(subset_texts, subset_emb, subset_ids)
        ram_after_build = check_process_ram()

        # Per-query latencies → p50/p95/p99.
        latencies_ms = []
        retrieved = []
        for qt, qe in zip(query_texts, query_embeddings):
            t0 = time.perf_counter()
            ret = retriever.search([qt], qe[None, :], k=10)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            retrieved.append(ret[0])
        p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])

        metrics = evaluate(eval_set, retrieved, ks=(1, 5, 10))

        results[size] = {
            "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)},
            "embeddings_plus_index_ram_mb": ram_after_build - ram_before,
            **metrics,
        }

    print("\n" + json.dumps(results, indent=2))

    csv_path = Path("results_hybrid_rrf.csv")
    fieldnames = [
        "size",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "embeddings_plus_index_ram_mb",
        "recall@1", "recall@5", "recall@10", "mrr@10",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for size, r in results.items():
            w.writerow({
                "size": size,
                "latency_p50_ms": r["latency_ms"]["p50"],
                "latency_p95_ms": r["latency_ms"]["p95"],
                "latency_p99_ms": r["latency_ms"]["p99"],
                "embeddings_plus_index_ram_mb": r["embeddings_plus_index_ram_mb"],
                "recall@1": r["recall@1"],
                "recall@5": r["recall@5"],
                "recall@10": r["recall@10"],
                "mrr@10": r["mrr@10"],
            })
    print(f"Wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
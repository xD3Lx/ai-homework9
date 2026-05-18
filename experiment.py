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

def main():
    CACHE_PATH = Path("template/cache/corpus.json")

    pool, eval_set = load_cache(CACHE_PATH)

    print("Loading embedding model...")
    # Use MPS on Apple Silicon
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    sizes = [1000, 10000, 100000, 300000]
    results = {}


if __name__ == "__main__":
    main()
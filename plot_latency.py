"""Plot p95 search latency vs corpus size for naive numpy vs FAISS HNSW."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    sizes = [int(r["size"]) for r in rows]
    p95 = [float(r["latency_p95_ms"]) for r in rows]
    return sizes, p95


def main():
    files = {
        "Naive NumPy (brute-force)": Path("results_naive.csv"),
        "FAISS HNSW":                Path("results_faiss_hnsw.csv"),
    }

    plt.figure(figsize=(8, 5))
    for label, path in files.items():
        if not path.exists():
            print(f"skipping missing file: {path}")
            continue
        sizes, p95 = load(path)
        plt.plot(sizes, p95, marker="o", label=label)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Corpus size (docs)")
    plt.ylabel("Search latency p95 (ms)")
    plt.title("Search latency p95 vs corpus size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out = Path("latency.png")
    plt.savefig(out, dpi=150)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

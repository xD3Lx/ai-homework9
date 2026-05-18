"""Plot recall@k comparison: naive dense vs hybrid (BM25 + dense + RRF)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    sizes = [int(r["size"]) for r in rows]
    return {
        "size":     sizes,
        "recall@1":  [float(r["recall@1"])  for r in rows],
        "recall@10": [float(r["recall@10"]) for r in rows],
    }


def find(*candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


def main():
    naive = find("results_naive.csv")
    hybrid = find("results_hybrid_rrf.csv")
    if not naive or not hybrid:
        print(f"missing CSVs (naive={naive}, hybrid={hybrid})")
        return

    n = load(naive)
    h = load(hybrid)

    metrics = ["recall@1", "recall@10"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, metrics):
        ax.plot(n["size"], n[m], marker="o", label="Naive dense")
        ax.plot(h["size"], h[m], marker="s", label="Hybrid (BM25+dense+RRF)")
        ax.set_xscale("log")
        ax.set_xlabel("Corpus size (docs)")
        ax.set_title(m)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Retrieval quality: naive dense vs hybrid RRF", y=1.02)
    fig.tight_layout()

    out = Path("recall.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

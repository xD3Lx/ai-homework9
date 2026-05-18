"""Create a retrieval quality plot from the baseline results CSV."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    source = Path("results_naive.csv")
    if not source.exists():
        raise FileNotFoundError(f"Missing input file: {source}")

    rows = load_rows(source)
    sizes = [int(row["size"]) for row in rows]
    recall_1 = [float(row["recall@1"]) for row in rows]
    recall_10 = [float(row["recall@10"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, recall_1, marker="o", label="Recall@1")
    plt.plot(sizes, recall_10, marker="s", label="Recall@10")
    plt.xscale("log")
    plt.ylim(0, 1.05)
    plt.xlabel("Corpus size (docs)")
    plt.ylabel("Score")
    plt.title("Baseline retrieval quality")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out = Path("plot.png")
    plt.savefig(out, dpi=150)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

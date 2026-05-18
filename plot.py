"""Create a retrieval quality plot from the baseline results CSV."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    """Read CSV rows from the given path."""
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    """Create a baseline Recall@1/Recall@10 plot and save it as plot.png."""
    source = Path("results_naive.csv")
    if not source.exists():
        raise FileNotFoundError(f"Missing input file: {source}")

    rows = load_rows(source)
    sizes, recall_1, recall_10 = [], [], []
    for idx, row in enumerate(rows, start=1):
        try:
            sizes.append(int(row["size"]))
            recall_1.append(float(row["recall@1"]))
            recall_10.append(float(row["recall@10"]))
        except (KeyError, ValueError) as err:
            raise ValueError(f"Invalid CSV data at row {idx}: {row}") from err

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

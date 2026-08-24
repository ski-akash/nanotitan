"""
Reads benchmarks/raw/results.csv (written by collect_results.py) and renders bar
charts comparing the bench_small_* configs on throughput, MFU, and peak memory.
Charts are saved as PNGs directly under benchmarks/ (committed, unlike raw/).

Usage: python benchmarks/plot_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARKS_DIR = Path(__file__).parent

CHARTS = [
    ("throughput(tps)", "Throughput (tokens/sec/device)", "throughput.png"),
    ("mfu(%)", "MFU (%)", "mfu.png"),
    ("memory/max_reserved(GiB)", "Peak reserved memory (GiB)", "memory.png"),
]


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def plot(rows: list[dict], key: str, title: str, out_name: str) -> None:
    configs = [r["config"] for r in rows]
    values = [float(r[key]) if r[key] else 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(configs, values)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(BENCHMARKS_DIR / out_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows(BENCHMARKS_DIR / "raw" / "results.csv")
    for key, title, out_name in CHARTS:
        plot(rows, key, title, out_name)
        print(f"wrote benchmarks/{out_name}")

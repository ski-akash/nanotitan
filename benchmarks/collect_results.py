"""
Pulls the bench_small_* runs' logged metrics back from WandB (that's the only place
MetricsProcessor writes them -- see nanotitan/components/metrics.py, there's no local
metrics file) and writes a summary table to benchmarks/summary.md plus the raw
per-run steady-state averages to benchmarks/raw/results.csv (gitignored).

Usage:
    python benchmarks/collect_results.py [--entity WANDB_TEAM] [--project nanotitan]
                                          [--skip-steps 5]

--skip-steps drops the first N logged rows of each run (warmup / compile /
first-step-slower-than-steady-state) before averaging the rest.
"""

import argparse
import csv
import os
from pathlib import Path

import wandb

BENCH_CONFIGS = [
    "bench_small_2gpu",
    "bench_small_ga4",
    "bench_small_ga8",
    "bench_small_dense",
    "bench_small_topk1",
    "bench_small_topk4",
]

METRIC_KEYS = [
    "throughput(tps)",
    "mfu(%)",
    "tflops",
    "time_metrics/end_to_end(s)",
    "time_metrics/data_loading(%)",
    "memory/max_reserved(GiB)",
]

BENCHMARKS_DIR = Path(__file__).parent


def collect(entity: str | None, project: str, skip_steps: int) -> list[dict]:
    api = wandb.Api()
    rows = []
    for run_name in BENCH_CONFIGS:
        path = f"{project}" if entity is None else f"{entity}/{project}"
        matches = api.runs(path, filters={"display_name": run_name})
        if len(matches) == 0:
            print(f"no wandb run found named '{run_name}' in {path}, skipping")
            continue
        run = matches[0]  # most recent if there are several

        history = run.history(keys=METRIC_KEYS, pandas=False)
        history = history[skip_steps:]
        if not history:
            print(f"run '{run_name}' has no logged rows after skip_steps, skipping")
            continue

        row = {"config": run_name}
        for key in METRIC_KEYS:
            values = [r[key] for r in history if key in r]
            row[key] = sum(values) / len(values) if values else None
        rows.append(row)

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict], out_path: Path) -> None:
    header = ["config", *METRIC_KEYS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [
            f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])
            for k in header
        ]
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=os.getenv("WANDB_TEAM"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "nanotitan"))
    parser.add_argument("--skip-steps", type=int, default=5)
    args = parser.parse_args()

    rows = collect(args.entity, args.project, args.skip_steps)
    write_csv(rows, BENCHMARKS_DIR / "raw" / "results.csv")
    write_summary_md(rows, BENCHMARKS_DIR / "summary.md")
    print(f"wrote {len(rows)} rows to benchmarks/raw/results.csv and benchmarks/summary.md")

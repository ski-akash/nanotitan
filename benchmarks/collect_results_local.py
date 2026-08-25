"""
Local-log fallback for collect_results.py, for use when no WANDB_API_KEY is
configured (trainer.slurm sets WANDB_MODE=offline in that case -- see its comment).
Parses MetricsProcessor's step log lines directly out of each config's most recent
outputs/<config>/*.err file instead of pulling from the WandB API, and writes the
same benchmarks/raw/results.csv + benchmarks/summary.md as collect_results.py.

Usage: python benchmarks/collect_results_local.py [--skip-steps 5] [--outputs-dir outputs]
"""

import argparse
import re
from pathlib import Path

from collect_results import BENCH_CONFIGS, write_csv, write_summary_md

BENCHMARKS_DIR = Path(__file__).parent

LINE_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[\d.]+)\s+grad_norm:\s*[\d.]+\s+"
    r"memory:\s*(?P<mem>[\d.]+)GiB\((?P<mem_pct>[\d.]+)%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+tflops:\s*(?P<tflops>[\d.]+)\s+mfu:\s*(?P<mfu>[\d.]+)%"
)

METRIC_KEYS = [
    "throughput(tps)",
    "mfu(%)",
    "tflops",
    "memory/max_reserved(GiB)",
]


def collect(outputs_dir: Path, skip_steps: int) -> list[dict]:
    rows = []
    for config in BENCH_CONFIGS:
        err_files = sorted(
            (outputs_dir / config).glob("*.err"), key=lambda p: p.stat().st_mtime
        )
        if not err_files:
            print(f"no .err logs found for '{config}' in {outputs_dir / config}, skipping")
            continue
        err_file = err_files[-1]  # most recent job

        parsed = [
            m.groupdict() for m in LINE_RE.finditer(err_file.read_text())
        ]
        if not parsed:
            print(f"no step lines found in {err_file}, skipping")
            continue
        parsed = parsed[skip_steps:]
        if not parsed:
            print(f"'{config}' has no rows left after skip_steps, skipping")
            continue

        row = {"config": config}
        row["throughput(tps)"] = sum(
            float(p["tps"].replace(",", "")) for p in parsed
        ) / len(parsed)
        row["mfu(%)"] = sum(float(p["mfu"]) for p in parsed) / len(parsed)
        row["tflops"] = sum(float(p["tflops"]) for p in parsed) / len(parsed)
        row["memory/max_reserved(GiB)"] = sum(
            float(p["mem"]) for p in parsed
        ) / len(parsed)
        row["time_metrics/end_to_end(s)"] = None
        row["time_metrics/data_loading(%)"] = None
        row["final_loss"] = float(parsed[-1]["loss"])
        rows.append(row)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    rows = collect(Path(args.outputs_dir), args.skip_steps)
    all_keys = [
        "throughput(tps)",
        "mfu(%)",
        "tflops",
        "time_metrics/end_to_end(s)",
        "time_metrics/data_loading(%)",
        "memory/max_reserved(GiB)",
        "final_loss",
    ]
    write_csv(rows, BENCHMARKS_DIR / "raw" / "results.csv", all_keys)
    write_summary_md(rows, BENCHMARKS_DIR / "summary.md", all_keys)
    print(f"wrote {len(rows)} rows to benchmarks/raw/results.csv and benchmarks/summary.md")

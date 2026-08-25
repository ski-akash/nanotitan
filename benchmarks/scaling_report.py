"""
B1 strong-scaling report (benchmarks/BENCHMARK_PLAN.md).

Parses the bench_scale_* SLURM logs and produces the scaling table: aggregate
throughput, speedup vs the 1-GPU baseline, scaling efficiency, and peak memory.

Every rank writes its own `step:` line to the job's .err, and `tps` in those lines
is *per device*, so rows are grouped by step, averaged across the ranks present,
and multiplied by world_size to get aggregate throughput. Steps at or below
--skip-steps are discarded as warmup; the rest give mean +/- stdev.

Usage: python benchmarks/scaling_report.py [--skip-steps 10] [--outputs-dir outputs]
"""

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent

# config name -> world_size
SCALE_POINTS = {
    "bench_scale_1gpu": 1,
    "bench_scale_2gpu": 2,
    "bench_scale_4gpu": 4,
}

LINE_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[\d.]+)\s+grad_norm:\s*[\d.]+\s+"
    r"memory:\s*(?P<mem>[\d.]+)GiB\((?P<mem_pct>[\d.]+)%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+tflops:\s*(?P<tflops>[\d.]+)\s+mfu:\s*(?P<mfu>[\d.]+)%"
)


def parse_run(err_file: Path, skip_steps: int):
    """-> (per-step aggregate-tps samples, peak memory GiB, final loss, n_ranks)"""
    by_step = defaultdict(list)
    mem_peak = 0.0
    last_loss = None
    for m in LINE_RE.finditer(err_file.read_text()):
        step = int(m["step"])
        by_step[step].append(float(m["tps"].replace(",", "")))
        mem_peak = max(mem_peak, float(m["mem"]))
        last_loss = float(m["loss"])

    ranks = max((len(v) for v in by_step.values()), default=0)
    samples = [
        statistics.fmean(tps_list)
        for step, tps_list in sorted(by_step.items())
        if step > skip_steps
    ]
    return samples, mem_peak, last_loss, ranks


def main(outputs_dir: Path, skip_steps: int) -> None:
    rows = []
    for config, world_size in SCALE_POINTS.items():
        err_files = sorted(
            (outputs_dir / config).glob("*.err"), key=lambda p: p.stat().st_mtime
        )
        if not err_files:
            print(f"[skip] no logs for {config}")
            continue

        # Pool every repeat of this scale point: the study calls for >=3 runs, and
        # variance across runs matters as much as variance across steps.
        samples, mem, loss, ranks, n_runs = [], 0.0, None, 0, 0
        for err_file in err_files:
            s, m, l, r = parse_run(err_file, skip_steps)
            if not s:
                continue
            samples += s
            mem = max(mem, m)
            loss = l if l is not None else loss
            ranks = max(ranks, r)
            n_runs += 1
        if not samples:
            print(f"[skip] {config}: no steps past warmup")
            continue

        # tps in the log is per device; aggregate over the whole job.
        per_dev = statistics.fmean(samples)
        agg = per_dev * world_size
        agg_sd = (statistics.stdev(samples) * world_size) if len(samples) > 1 else 0.0
        rows.append(
            {
                "config": config,
                "world_size": world_size,
                "ranks_logging": ranks,
                "per_device_tps": per_dev,
                "aggregate_tps": agg,
                "aggregate_sd": agg_sd,
                "peak_mem_gib": mem,
                "final_loss": loss,
                "n_samples": len(samples),
                "n_runs": n_runs,
            }
        )

    if not rows:
        print("no scaling data found")
        return

    baseline = next((r for r in rows if r["world_size"] == 1), None)
    for r in rows:
        if baseline:
            r["speedup"] = r["aggregate_tps"] / baseline["aggregate_tps"]
            r["efficiency"] = 100 * r["speedup"] / r["world_size"]
        else:
            r["speedup"] = r["efficiency"] = float("nan")

    hdr = f"{'GPUs':>4}  {'aggregate tok/s':>20}  {'speedup':>8}  {'efficiency':>10}  {'peak mem':>9}  {'runs':>4}  {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        agg = f"{r['aggregate_tps']:,.0f} +/- {r['aggregate_sd']:,.0f}"
        print(
            f"{r['world_size']:>4}  {agg:>20}  {r['speedup']:>7.2f}x  "
            f"{r['efficiency']:>9.1f}%  {r['peak_mem_gib']:>7.2f}G  {r['n_runs']:>4}  {r['n_samples']:>4}"
        )

    lines = [
        "| GPUs | scope | aggregate tokens/sec | speedup | scaling efficiency | peak mem/GPU | runs |",
        "|---|---|---|---|---|---|---|",
    ]
    scope = {
        1: "single GPU",
        2: "intra-node (PCIe `SYS`)",
        4: "**inter-node (1 GbE)**",
    }
    for r in rows:
        lines.append(
            f"| {r['world_size']} | {scope.get(r['world_size'], '')} | "
            f"{r['aggregate_tps']:,.0f} ± {r['aggregate_sd']:,.0f} | "
            f"{r['speedup']:.2f}× | {r['efficiency']:.1f}% | "
            f"{r['peak_mem_gib']:.2f} GiB | {r['n_runs']} |"
        )
    out = BENCHMARKS_DIR / "raw" / "scaling.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()
    main(Path(args.outputs_dir), args.skip_steps)

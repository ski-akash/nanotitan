#!/bin/bash
# Submits the 2x A100 benchmark configs (bench_small_*, defined in
# nanotitan/config/default_configs.py) as SLURM jobs via launch.sh.
#
# Usage: PARTITION=gpu-A100 ./benchmarks/run.sh [config_name ...]
#   No args -> submits every bench_small_* config.
#   Args    -> submits only the named configs (must exist in config_map).
#
# Each config uses data_parallel_replicate_degree=2, so it needs exactly one
# singlenode (2-GPU) allocation -- see launch.sh/trainer.slurm.

set -e

cd "$(dirname "$0")/.."

DEFAULT_CONFIGS=(
    bench_small_2gpu
    bench_small_ga4
    bench_small_ga8
    bench_small_dense
    bench_small_topk1
    bench_small_topk4
)

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

for cfg in "${CONFIGS[@]}"; do
    ./launch.sh singlenode "$cfg"
done

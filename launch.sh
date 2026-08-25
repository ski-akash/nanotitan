#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <singlenode|multinode|NUM_NODES> <config_name> [extra args...]"
    echo "  Each node on the gpu-A100/gpu-P100 partitions has 2 GPUs, so by default:"
    echo "    singlenode -> 1 node  -> 2 GPUs"
    echo "    multinode  -> 2 nodes -> 4 GPUs"
    echo
    echo "  GPUS_PER_NODE overrides GPUs per node (default 2), for the scaling study:"
    echo "    GPUS_PER_NODE=1 $0 singlenode bench_scale_1gpu   # world_size 1"
    echo "                     $0 singlenode bench_scale_2gpu   # world_size 2"
    echo "                     $0 multinode  bench_scale_4gpu   # world_size 4"
    echo
    echo "  The config's parallelism degrees must multiply to the resulting"
    echo "  world_size (nodes x GPUS_PER_NODE) or ParallelDims will reject it."
    exit 1
fi

case $1 in
    singlenode) NODES=1 ;;
    multinode)  NODES=2 ;;
    *[!0-9]*)   echo "Error: first argument must be singlenode, multinode, or a number"; exit 1 ;;
    *)          NODES=$1 ;;
esac

CONFIG=$2
shift 2

# Which SLURM partition to submit to: gpu-A100 (default) or gpu-P100 for
# occasional P100 runs. Override with PARTITION=gpu-P100 ./launch.sh ...
PARTITION="${PARTITION:-gpu-A100}"

# GPUs per node. Both partitions have 2 per node, which is the default; the
# scaling study (benchmarks/BENCHMARK_PLAN.md, B1) needs a 1-GPU point too.
# Passed to sbatch explicitly so it overrides trainer.slurm's #SBATCH directive,
# and exported so trainer.slurm can match --nproc_per_node to it.
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

if [ "$CONFIG" = "all" ]; then
    CONFIGS=$(grep -oP '^\s+"\K[^"]+(?=":)' nanotitan/config/default_configs.py)
    echo "Will schedule configs: $CONFIGS"
    echo "This will delete the outputs/ folder."
    read -p "Proceed? [y/N] " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf outputs
    for cfg in $CONFIGS; do
        mkdir -p outputs/$cfg
        export WANDB_RUN_NAME="$cfg"
        sbatch --job-name="$cfg" --partition="$PARTITION" --nodes=$NODES --gpus-per-node=$GPUS_PER_NODE trainer.slurm "$@"
    done
else
    mkdir -p outputs/$CONFIG
    export WANDB_RUN_NAME="$CONFIG"
    sbatch --job-name="$CONFIG" --partition="$PARTITION" --nodes=$NODES --gpus-per-node=$GPUS_PER_NODE trainer.slurm "$@"
fi

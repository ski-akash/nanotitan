#!/bin/bash

#SBATCH --chdir=/userhome/mtech/akashc1005/nanotitan/
#SBATCH --cpus-per-task=16
#SBATCH --error=/userhome/mtech/akashc1005/nanotitan/outputs/%x/%j.err
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=60G
#SBATCH --output=/userhome/mtech/akashc1005/nanotitan/outputs/%x/%j.out
# --partition is passed explicitly by launch.sh (gpu-A100 by default, gpu-P100 for
# occasional P100 runs); no default here so a bare `sbatch trainer.slurm` fails loudly
# instead of silently landing on the wrong hardware.

cd "$TMPDIR" || exit 1
cd /userhome/mtech/akashc1005/nanotitan/ || exit 1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# Multi-node comms. csecluster has no InfiniBand/RDMA -- inter-node traffic goes
# over plain 1GbE on eno5 (eno7/eno8/ens7f0/ens7f1 exist but are DOWN). Without
# pinning the interface, NCCL and the c10d rendezvous can pick a down NIC and
# hang instead of failing. Harmless on single-node jobs.
export NCCL_SOCKET_IFNAME=eno5
export GLOO_SOCKET_IFNAME=eno5
export NCCL_IB_DISABLE=1

# No WandB credentials configured on csecluster -- log locally instead of failing
# the job outright. Sync later with `wandb sync` once a WANDB_API_KEY is set up.
export WANDB_MODE=offline

export NANOTITAN_CONFIG="$SLURM_JOB_NAME"

mkdir -p outputs/$NANOTITAN_CONFIG

# Each node on gpu-A100/gpu-P100 has 2 GPUs, which is the default. launch.sh
# exports GPUS_PER_NODE (and passes the matching --gpus-per-node to sbatch) so the
# scaling study can also run a 1-GPU point; world_size = nnodes x GPUS_PER_NODE.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
echo "world_size = $SLURM_JOB_NUM_NODES nodes x $GPUS_PER_NODE gpus = $(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))"

srun uv run torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node $GPUS_PER_NODE --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m nanotitan.train "$@"

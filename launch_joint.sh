#!/bin/bash
# Joint 2-user, 4-GPU benchmark launcher.
#
# Prereq: each user is capped at 2 GPUs on this cluster, so a single sbatch job
# spanning --nodes=2 under one account will not get you 4 GPUs. Instead, each
# user grabs their OWN 1-node/2-GPU SLURM allocation, and both sides join the
# same torchrun rendezvous manually. This requires the two allocations' nodes
# to be able to reach each other over TCP -- not guaranteed on every cluster;
# if rendezvous just hangs, that's likely why (ask your admin, or fall back to
# a 2-GPU benchmark).
#
# Usage:
#   Host  (runs this repo, config authority): ./launch_joint.sh host <config>
#   Peer  (your friend, joining):              ./launch_joint.sh peer <config> <host_ip>
#
# Both commands must be run at roughly the same time, inside a salloc'd
# allocation (see instructions printed below), not via sbatch.

set -e

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  Host: $0 host <config_name>"
    echo "  Peer: $0 peer <config_name> <host_ip>"
    exit 1
fi

ROLE=$1
CONFIG=$2
PARTITION="${PARTITION:-gpu-A100}"
RDZV_ID=101
RDZV_PORT=29500

if [ "$ROLE" = "host" ]; then
    echo "== HOST =="
    echo "1) Get an interactive 1-node/2-GPU allocation:"
    echo "     salloc --partition=$PARTITION --nodes=1 --gpus-per-node=2 --cpus-per-task=16 --mem-per-gpu=60G"
    echo "2) Once granted, find this node's IP and share it with your peer:"
    echo "     hostname --ip-address"
    echo "3) Then run (inside the salloc shell, from the repo root):"
    echo "     mkdir -p outputs/$CONFIG"
    echo "     NANOTITAN_CONFIG=$CONFIG uv run torchrun --nnodes 2 --node_rank 0 --nproc_per_node 2 \\"
    echo "         --rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint \$(hostname --ip-address):$RDZV_PORT \\"
    echo "         -m nanotitan.train"
    echo
    echo "Wait for your peer to launch their side within ~60s of this (rdzv will time out otherwise)."

elif [ "$ROLE" = "peer" ]; then
    HOST_IP=$3
    if [ -z "$HOST_IP" ]; then
        echo "Peer needs the host's IP: $0 peer $CONFIG <host_ip>"
        exit 1
    fi
    echo "== PEER =="
    echo "1) Get an interactive 1-node/2-GPU allocation on YOUR account:"
    echo "     salloc --partition=$PARTITION --nodes=1 --gpus-per-node=2 --cpus-per-task=16 --mem-per-gpu=60G"
    echo "2) Once granted, run (inside the salloc shell, from the repo root -- same code/config as host):"
    echo "     NANOTITAN_CONFIG=$CONFIG uv run torchrun --nnodes 2 --node_rank 1 --nproc_per_node 2 \\"
    echo "         --rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint $HOST_IP:$RDZV_PORT \\"
    echo "         -m nanotitan.train"
    echo
    echo "Launch this within ~60s of the host's command, or rdzv will time out."
else
    echo "First arg must be 'host' or 'peer'"
    exit 1
fi

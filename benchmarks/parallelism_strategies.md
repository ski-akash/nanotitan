# B5 — DDP vs FSDP vs HSDP at fixed 4-GPU scale

Experiment B5 from `BENCHMARK_PLAN.md`. Same 163M model, `seq_len` 1024, global batch
16, `ga=1`, `seed=42`, 15 steps, `world_size=4` (2 nodes x 2 GPUs) in every run. Only
the arrangement of the 4 ranks changes. Configs: `bench_par_{ddp,fsdp,hsdp}_4gpu`.

Note the reference's own `ddp`/`fsdp`/`hsdp` presets assume a 16-GPU cluster
(`dp_shard=16`) and cannot run here; these are sized for 4.

## Result

| strategy | layout | aggregate tok/s | vs DDP | peak mem/GPU | vs DDP |
|---|---|---|---|---|---|
| DDP | `dp_replicate=4` | **2,836** | — | 9.02 GiB | — |
| FSDP | `dp_shard=4` | 2,072 | −26.9% | **6.47 GiB** | −28.3% |
| HSDP | `2 shard x 2 replicate` | 2,756 | −2.8% | 6.81 GiB | −24.5% |

## Reading it

**HSDP strictly dominates FSDP on this cluster** — 33% more throughput for only
0.34 GiB more memory. There is no configuration here in which FSDP is the right
choice over HSDP.

**HSDP is the sensible default overall**: it gives up 2.8% throughput (within a
couple of standard deviations of DDP) for 24.5% less memory. DDP wins only if memory
is free, which it is not once the model grows.

**Why FSDP loses throughput.** DDP issues a single gradient all-reduce per step,
which overlaps with backward compute. FSDP issues three collectives — two parameter
all-gathers and a reduce-scatter — and the all-gathers sit on the *forward* critical
path where there is no backward work to hide them behind. On a 1 GbE link with no
RDMA, exposed round trips are expensive, so more collectives costs more than the
slightly smaller total volume saves.

**Why HSDP nearly matches DDP.** Its 2x2 layout shards within a node (over PCIe) and
replicates across the 1 GbE boundary, so the inter-node collective carries sharded
(half-size) gradients. The saving on the wire is largely cancelled by the extra
intra-node reduce-scatter/all-gather, leaving throughput about level.

## Memory: the saving is larger than state-sharding arithmetic predicts

Weights + gradients + AdamW state for this model at fp32 is ~2.6 GB, so naive
sharding math predicts HSDP (2-way) saves ~1.2 GiB and FSDP (4-way) ~1.8 GiB versus
DDP. Measured savings are **2.21 GiB and 2.55 GiB** — both larger.

The excess comes from two effects the arithmetic misses: `apply_fsdp` sets
`param_dtype=bfloat16`, so the sharded parameter copies used during compute are
half-width, and `reshard_after_forward` frees gathered parameters between forward and
backward rather than holding them.

Note also that FSDP shards 4-way versus HSDP's 2-way but saves only 0.34 GiB more.
At this model size **activations dominate the footprint**, and no data-parallel
sharding strategy touches activations — that is what activation checkpointing (B6)
is for. Sharding pays off proportionally more as the model grows, since parameter
state scales with parameter count while activations scale with batch x sequence.

## Prediction vs. measurement

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| FSDP throughput vs DDP | roughly level (volume ~equal) | −26.9% | ❌ missed |
| FSDP memory saving | ~2 GiB | 2.55 GiB | ✅ close |
| HSDP throughput vs DDP | **faster** (halved inter-node bytes) | −2.8% | ❌ missed |
| HSDP memory | ~7.7 GiB | 6.81 GiB | ❌ saved more than expected |

Two lessons from the misses, both the same shape as B4b's: reasoning about
communication **volume** alone predicts poorly. What actually governed the result was
the *number of exposed collectives* and *where they sit relative to compute* — FSDP's
forward-path all-gathers cannot overlap, which volume math does not capture. And
memory models that count only optimizer state miss dtype policy and resharding
behaviour.

## Caveat

One run per strategy (n=1). Throughput at 4 GPUs was highly reproducible in B1
(±1.0%, since the runs are communication-bound and therefore deterministic), so the
26.9% FSDP gap is far outside noise and is safe to state. The 2.8% DDP-vs-HSDP
difference is **not** safely outside noise and should be described as "about level"
rather than as DDP winning, unless repeats are run.

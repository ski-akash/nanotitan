# B1 — Strong-scaling study (1 / 2 / 4 × A100)

Experiment B1 from `BENCHMARK_PLAN.md`. Model: 163M-param DeepSeek-V3
(MLA + MoE, top_k=2 of 8). DDP, fp32 master weights with bf16 autocast,
`torch.compile` off. **Strong scaling**: global batch 16, `seq_len` 1024 and 40
steps are fixed at every point, so all three configs process an identical
655,360 tokens and differ only in how many GPUs divide the work.
`gradient_accumulation_steps == 1` throughout, so this isolates scaling.

3 repeats per point; 30 post-warmup steps each (first 10 discarded) = 90 samples.
Reproduce: `GPUS_PER_NODE=1 ./launch.sh singlenode bench_scale_1gpu`,
`./launch.sh singlenode bench_scale_2gpu`, `./launch.sh multinode bench_scale_4gpu`,
then `python benchmarks/scaling_report.py`.

## Result

| GPUs | scope | aggregate tokens/sec | speedup | scaling efficiency | peak mem/GPU |
|---|---|---|---|---|---|
| 1 | single GPU | 50,943 ± 5,211 | 1.00× | 100% | 24.16 GiB |
| 2 | intra-node (PCIe `SYS`, no NVLink) | 50,828 ± 4,470 | 1.00× | **49.9%** | 13.07 GiB |
| 4 | **inter-node (1 GbE, no RDMA)** | 2,836 ± 28 | 0.06× | **1.4%** | 9.02 GiB |

**Adding a second GPU produced no speedup whatsoever** (50,943 → 50,828 tok/s), and
four GPUs run **18× slower than one**.

## Step-time decomposition

Compute time per step is estimated from the 1-GPU baseline scaled by local batch
(compute is proportional to local work); communication is the remainder.

| GPUs | step time | est. compute | est. comm | **comm fraction** |
|---|---|---|---|---|
| 1 | 322 ms | 322 ms | — | 0% |
| 2 | 322 ms | 161 ms | 161 ms | **50.1%** |
| 4 | 5,777 ms | 80 ms | 5,697 ms | **98.6%** |

At 2 GPUs communication and computation are almost exactly equal, so halving the
per-device work is precisely cancelled by the cost of synchronising it — the
textbook Amdahl case, landing on 50% efficiency for a structural reason rather than
by coincidence.

## Implied interconnect bandwidth

The model holds **652 MB of fp32 gradients**. A ring all-reduce moves
`2(N−1)/N × 652 MB` per rank.

- **2 GPUs (intra-node):** 652 MB ÷ 161 ms ⇒ **≈ 4.0 GB/s**. Consistent with `SYS`
  topology — no NVLink and no P2P, so traffic goes GPU→host→UPI→host→GPU.
- **4 GPUs (inter-node):** 978 MB of ring traffic ÷ 5.697 s ⇒ ≈ 172 MB/s overall.
  Two of the four ring links cross the node boundary, so roughly half that traffic
  is on the wire: **≈ 86 MB/s, about 69% of 1 GbE line rate** — a realistic TCP
  efficiency, confirming the 1 GbE link is the binding constraint.

## Prediction vs. measurement

Recorded in `BENCHMARK_PLAN.md` **before** these jobs were submitted:

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| 4-GPU step time | ~6 s | 5.78 s | ✅ within 4% |
| 2-GPU efficiency | 75–95% | 49.9% | ❌ far too optimistic |
| 4-GPU efficiency | 5–20% | 1.4% | ❌ too optimistic |

The **bandwidth-derived timing model was accurate**; the efficiency predictions were
not. The error was assuming intra-node communication would be cheap. It is not:
`SYS` topology gives ~4 GB/s, and against a model this small (161 ms of compute per
step at 2 GPUs) that is enough to consume the entire benefit of the second GPU. The
lesson is that the compute-to-communication ratio, not the GPU count, determines
whether scaling is possible at all.

## Corroboration

Variance is itself evidence. Relative standard deviation is **±10.2%** at 1 GPU and
**±8.8%** at 2 GPUs, but only **±1.0%** at 4 GPUs. A compute-bound run varies with
GC, data loading and kernel scheduling; a run that is 98.6% blocked on a saturated
1 GbE link is essentially deterministic. All four ranks reported an identical
708 tok/s on nearly every step.

## What this implies

1. **This cluster cannot data-parallel-scale this model as configured.** That is a
   property of the compute:communication ratio, not a bug.
2. **Communication volume and frequency are the only levers that matter here.**
   Which makes the pending experiments the important ones: working gradient
   accumulation (B4b — currently every microbatch triggers a full all-reduce, see
   `BENCHMARK_PLAN.md` §2), bucketing (Tier 4), and FSDP's sharded/bf16 collectives
   versus DDP's fp32 all-reduce (B5).
3. **Scaling would improve with a larger model**, which raises compute per step
   while gradient volume grows only linearly — the 2-GPU point should improve
   materially before the 4-GPU point does.

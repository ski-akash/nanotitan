# Benchmark results

All runs: 2x A100 (PCIe, 40GB), DDP (`data_parallel_replicate_degree=2`), fp32
master weights with bf16 autocast, `torch.compile` disabled (no `python3-dev` on
csecluster's compute nodes). Numbers are steady-state averages (first 5 logged rows
skipped as warmup), **single run per config -- no repeats, so no variance is
reported**. Raw per-run data: `benchmarks/raw/results.csv`.

> **Scope note.** Every run here is at a single scale (2 GPUs), so this file
> contains *no scaling result*. The scaling study, communication analysis, and the
> methodology these runs are missing (repeats, equal-token comparisons) are
> specified in `benchmarks/BENCHMARK_PLAN.md`. Read that before quoting anything
> here. MFU of 4-9% is a real measurement but reflects a small uncompiled model,
> not a tuned system -- treat it as a diagnostic, not a headline.

## 1. DDP baseline + gradient-accumulation ablation

163M-param model, `seq_len=2048`, `local_batch_size=8`, 200 steps.

| config | grad-accum steps | tokens seen | tokens/sec/device | MFU | peak memory | final loss |
|---|---|---|---|---|---|---|
| `bench_small_2gpu` | 1 | 6.6M | 41,987 | 7.57% | 24.65 GiB (62%) | 6.945 |
| `bench_small_ga4` | 4 | 26.2M | 44,470 | 8.02% | 24.67 GiB (62%) | (6.586) |
| `bench_small_ga8` | 8 | 52.4M | 44,903 | 8.10% | 24.67 GiB (62%) | (6.396) |

**Reading it**: throughput rises slightly with more accumulation. Note this is a
*smaller* gain than expected -- see the `no_sync` finding in `BENCHMARK_PLAN.md`
§2, which explains why: gradient accumulation is not currently suppressing the
per-microbatch all-reduce, so it is not reducing communication.

⚠️ **The loss column is not a valid comparison and is parenthesised for that
reason.** All three ran 200 steps, but global batch size scales with the
accumulation factor, so `ga8` saw 8× more tokens than the baseline. Lower loss is
arithmetically expected and says nothing about gradient accumulation. A valid
comparison requires equal tokens seen (planned as experiment B4).

## 2. MoE ablation

Same 163M-param model/steps, varying only the mixture-of-experts routing (baseline
uses `top_k=2` of 8 experts).

| config | routing | tokens/sec/device | MFU | peak memory | final loss |
|---|---|---|---|---|---|
| `bench_small_dense` | no MoE (all layers dense) | 51,007 | 9.20% | 24.53 GiB (62%) | 6.917 |
| `bench_small_topk1` | 1 of 8 experts active | 43,355 | 7.36% | 24.62 GiB (62%) | 6.935 |
| `bench_small_2gpu` | 2 of 8 experts active (baseline) | 41,987 | 7.57% | 24.65 GiB (62%) | 6.945 |
| `bench_small_topk4` | 4 of 8 experts active | 40,914 | 8.25% | 24.87 GiB (62%) | 6.942 |

**Reading it**: dense is fastest (~22% higher throughput than 2-expert MoE) since
there's no router/dispatch overhead. Throughput drops as more experts activate per
token (more active compute), roughly as expected; memory stays flat since all 8
experts are always resident regardless of how many are active per token.

## 3. Model-size sanity check

Same 2-GPU DDP setup, no MoE/grad-accum variation -- just confirming a much bigger
model still fits and trains stably.

| config | params | seq_len | batch | steps | tokens/sec/device | MFU | peak memory | final loss |
|---|---|---|---|---|---|---|---|---|
| `bench_small_2gpu` | 163M | 2048 | 8 | 200 | 41,987 | 7.57% | 24.65 GiB (62%) | 6.945 |
| `bench_1b_2gpu` | 976M | 1024 | 4 | 50 | 4,484 | 4.21% | 23.12 GiB (58%) | 7.481 |

**Reading it**: the 976M model runs stably with real headroom left (23/40 GiB) even
with DDP replicating the full model per GPU and no sharding -- there's room to grow
batch size or sequence length before hitting memory limits. Throughput is ~10x lower
per token, roughly tracking the ~6x parameter increase plus no `torch.compile`. MFU
is lower too (4.2% vs 7.6%) -- expected without compile-driven kernel fusion.

⚠️ **`bench_1b_2gpu` is not a trained model and must never be described as one.**
It ran 50 steps × global batch 8 × seq_len 1024 = **~0.4M tokens** -- roughly six
orders of magnitude short of what pretraining a ~1B model requires. Its only purpose
is to confirm a model of that size fits in 40GB and trains without diverging or
OOMing.

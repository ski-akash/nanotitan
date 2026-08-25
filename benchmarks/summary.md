# Benchmark results

All runs: 2x A100 (PCIe, 40GB), DDP (`data_parallel_replicate_degree=2`), fp32,
`torch.compile` disabled (no `python3-dev` on csecluster's compute nodes). Numbers
are steady-state averages (first 5 logged rows skipped as warmup). Raw per-run data:
`benchmarks/raw/results.csv`.

## 1. DDP baseline + gradient-accumulation ablation

163M-param model, `seq_len=2048`, `local_batch_size=8`, 200 steps.

| config | grad-accum steps | tokens/sec/device | MFU | peak memory | final loss |
|---|---|---|---|---|---|
| `bench_small_2gpu` | 1 | 41,987 | 7.57% | 24.65 GiB (62%) | 6.945 |
| `bench_small_ga4` | 4 | 44,470 | 8.02% | 24.67 GiB (62%) | 6.586 |
| `bench_small_ga8` | 8 | 44,903 | 8.10% | 24.67 GiB (62%) | 6.396 |

**Reading it**: more gradient accumulation gave both slightly higher throughput
(fewer, larger sync points amortize the DDP all-reduce cost) and a lower loss after
the same 200 steps (larger effective batch → more stable gradient estimate).

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

**Reading it**: the 1B model trains stably with real headroom left (23/40 GiB) even
at fp32 with no sharding -- there's room to grow batch size or sequence length
before hitting memory limits. Throughput is ~10x lower per token, roughly tracking
the ~6x parameter increase plus no `torch.compile`. MFU is lower too (4.2% vs 7.6%)
-- expected without compile-driven kernel fusion.

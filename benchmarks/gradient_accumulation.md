# B4b — Gradient accumulation and the missing `no_sync`

Experiment B4b from `BENCHMARK_PLAN.md`. Same 163M model, `seq_len` 1024 and
per-device microbatch as the B1 scaling study, so numbers are directly comparable.
`ga=8`: eight microbatches per optimizer step. All runs `seed=42`, 12 steps.

## The defect

`Trainer.train_step` ran its microbatch loop with no gradient-sync control, and the
reference (`torchfeather`) has none anywhere either — verified by grep across the
whole package. Under DDP that means **every microbatch triggers a full gradient
all-reduce**, so gradient accumulation raised the effective batch size without
reducing communication at all.

The fix suppresses the sync on all microbatches but the last via
`set_requires_gradient_sync(False)`, so gradients accumulate locally and synchronise
once per optimizer step. It is a **deliberate, documented divergence from the
reference**, marked as such in `train.py`.

## Correctness first

The change is mathematically neutral, not an approximation: the collective computes
a mean over ranks, and `Σᵢ mean(gradᵢ) == mean(Σᵢ gradᵢ)`. Only communication volume
changes. Verified empirically with a fixed seed — loss and gradient norm are
**bit-identical** before and after:

| 4-GPU `ga=8` | loss @ step 1 | @ step 12 | grad_norm @ 12 |
|---|---|---|---|
| before fix | 12.0034 | 10.7031 | 0.3382 |
| after fix | 12.0034 | 10.7031 | 0.3382 |

## Result

| scale | before fix | after fix | change |
|---|---|---|---|
| 2 GPU (intra-node, PCIe `SYS`) | 72,102 tok/s | 71,916 tok/s | **none (within noise)** |
| 4 GPU (inter-node, 1 GbE) | 2,868 tok/s | **18,724 tok/s** | **6.5×** |

**The fix matters exactly where communication is exposed, and not at all where it is
already hidden.**

- At **2 GPUs** DDP's bucketed all-reduce already overlaps with backward compute, so
  there was little exposed communication left to remove. Suppressing 7 of 8 syncs
  changes nothing measurable.
- At **4 GPUs** the all-reduce costs ~5.7 s against ~80 ms of compute (B1). No amount
  of overlap can hide a collective 70× longer than the compute it would overlap
  with, so removing 7 of every 8 of them converts almost directly into throughput.

## Scaling efficiency, combining accumulation and the fix

Measured against a 1-GPU `ga=8` baseline — using the `ga=1` baseline instead would
overstate the gain, because accumulation also amortises per-optimizer-step overhead
(the AdamW update and the grad-norm all-reduce for clipping) on a single GPU too.

| GPUs | B1 (`ga=1`) | B4b (`ga=8` + fix) |
|---|---|---|
| 2 | 49.9% | **63.5%** |
| 4 | 1.4% | **8.3%** |

## Predictions vs. measurement

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| 4-GPU control (before fix, `ga=8`) | ~709 tok/s/dev | 717 | ✅ within 1% |
| 4-GPU after fix | ~5,221 tok/s/dev | 4,681 | ✅ within 11% |
| 2-GPU efficiency after fix | ~89% | 63.5% | ❌ overestimated |

The 2-GPU projection assumed the eight all-reduces were paid serially. They were not
— DDP had already overlapped most of that cost, so the volume-based estimate
overstated the available saving. **Volume-based communication models overestimate
the benefit of any optimisation wherever the communication is already overlapped**;
they are only predictive when the collective is exposed, as at 4 GPUs.

## Method note — a corrected error

An earlier reading of this experiment reported "+21.9% at 2 GPUs" from the fix. That
was wrong: it compared a single unseeded run against another single unseeded run,
and the apparent gain was a cold-start outlier (30,037 tok/s on the first run after a
code sync, versus ~36,000 on every subsequent run). Repeating with a fixed seed
showed before and after are identical at 2 GPUs. The claim is withdrawn.

Two process failures caused it, both now fixed: `n=1` comparisons in violation of
this study's own ≥3-repeat rule, and `training.seed` left at its `None` default so
runs were not reproducible. The bench configs now set `seed=42`.

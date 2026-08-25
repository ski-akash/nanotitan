# Benchmark plan — nanotitan on csecluster

Operational spec for the benchmarking study. Read this before running or changing
anything under `benchmarks/`. Written after auditing the first round of runs
(commit `8c1cf8b`) against what the hardware can actually demonstrate.

---

## 1. Hardware reality (measured, not assumed)

Everything below was measured on `gpu-A100-02` via `srun`, not taken from docs.

| Property | Value | Consequence |
|---|---|---|
| GPUs | 4× A100-**PCIe-40GB** (2 nodes × 2) | 40GB, not 80GB — caps model size |
| Intra-node GPU link | **`SYS`** (PCIe + UPI across NUMA) | No NVLink, no P2P bridge — worst-case intra-node |
| Inter-node link | **1 GbE** (`eno5`, ~110 MB/s effective) | No InfiniBand, no RDMA |
| SLURM QOS | `MaxSubmitPU=2` | Never queue more than 2 jobs |
| Compute-node internet | none | Dataset must be local (already handled) |
| `python3-dev` on compute nodes | absent | `torch.compile` currently disabled |

**This defines the whole study.** These are not limitations to apologise for — they
make the cluster an unusually *clear* subject for a communication-bottleneck
analysis. On a DGX with NVLink + InfiniBand, communication effects hide behind fast
links and every optimisation looks marginal. Here they are the dominant term and
every optimisation is measurable.

The number that matters: the 163M model holds **652 MB of fp32 gradients**
(163e6 × 4 B). A ring all-reduce moves ≈`2(N-1)/N × 652 MB` per rank per sync.
Over a 1 GbE link at ~110 MB/s, one such all-reduce costs **~6 seconds**. Measured
2-GPU step time is ~0.39 s. That single ratio predicts the entire scaling story
before a job is even submitted — and stating a prediction *before* measuring, then
checking it, is the difference between a benchmark and a systems study.

---

## 2. Audit of what has been done

### Built (this part is genuinely strong)

A from-scratch port of a DeepSeek-V3-style distributed training framework: MLA
attention, MoE with token-choice routing and grouped GEMM, RoPE/YaRN, and the full
parallelism stack (DDP, FSDP, HSDP, TP, PP, EP, CP) behind one `DeviceMesh`
abstraction, plus checkpointing, metrics/MFU, and profiling hooks. ~7 commits of
incremental, reviewable work. Seven benchmark configs ran end to end on real
hardware after solving four independent cluster-environment blockers (job-submit
policy, no compute-node internet, no WandB key, no compiler headers).

### Measured (this part is thin)

Seven runs, all at **one scale (2 GPUs)**, one repeat each. That yields comparisons
between configs but **no scaling result at all** — and scaling is the entire point
of a distributed training framework.

### Two methodological defects to fix before any of this is quoted

**(a) The gradient-accumulation loss comparison in `summary.md` is confounded.**
`bench_small_ga8` reached loss 6.40 vs baseline 6.95 — but at equal *step* count
with 8× the global batch, it saw **8× more tokens**. Lower loss is arithmetically
guaranteed and says nothing about gradient accumulation. Loss must only be compared
at **equal tokens seen**. (The throughput comparison in that table is sound; only
the loss column is invalid.)

**(b) MFU of 4–9% is real but must not be quoted as a headline.** The denominator
(312 TFLOPS, A100 dense bf16) is correct and autocast bf16 *is* active on the DDP
path, so the measurement is honest. It is low because the model is small
(dim=512), `torch.compile` is off, and per-step work is small — all explicable, none
impressive. An interviewer will ask "why is your MFU 8%?" and the honest answer
undercuts the bullet. **Report MFU as a diagnostic, and make scaling efficiency and
communication analysis the headline instead** — those numbers are strong and
defensible on this hardware.

### One substantive finding already surfaced by the audit

`Trainer.train_step` (`nanotitan/train.py:399`) runs its gradient-accumulation
microbatch loop with **no `no_sync()` / `set_requires_gradient_sync(False)` guard**.
Under DDP every microbatch therefore triggers a full gradient all-reduce, so
gradient accumulation currently increases effective batch size **without reducing
communication** — which is its main value on a bandwidth-starved cluster. This is
consistent with the measured data: ga8 gave only ~6% throughput gain where
suppressing 7 of every 8 all-reduces should have given far more.

On a 1 GbE cluster this is the single highest-impact fix available. **It modifies
reference-ported code, so it needs your sign-off** (project rule: no inventing —
check `reference/torchfeather` first to confirm this is a genuine gap rather than
something handled elsewhere). Treat it as experiment **B4b** below.

---

## 3. The benchmark study

Ordered by (resume value × feasibility). **Tier 1 is mandatory** — without it there
is no distributed-systems result. Tiers 2–3 are the optimisation narrative.

Standing method for every experiment: **≥3 repeats**, report **mean ± stdev**,
discard the first **10 steps** as warmup, fixed seed, record git SHA + torch version
+ `nvidia-smi topo -m` alongside results, and compare loss only at equal tokens seen.

### Tier 1 — Strong scaling and the communication bottleneck (mandatory)

**B1. Strong-scaling study: 1 → 2 → 4 GPUs.**
Fixed *global* batch size and fixed total tokens; vary GPU count. Report tokens/sec
(aggregate), speedup vs 1 GPU, **scaling efficiency %**, and peak memory/GPU.
- 1 GPU: `dp_replicate=1`, 1 node — requires parameterising `--nproc_per_node`
- 2 GPU: `dp_replicate=2`, 1 node — intra-node PCIe/`SYS`
- 4 GPU: `dp_replicate=4`, 2 nodes — **crosses the 1 GbE boundary**

*Predict before running:* 2-GPU efficiency 75–95%; 4-GPU efficiency 5–20%. If
4-GPU efficiency comes out >50%, do not celebrate — verify the job genuinely spanned
two nodes and that NCCL did not silently fall back.

Use ≤50 steps for 4-GPU points (a step may take seconds) — it holds *both* nodes, so
it is the highest-contention experiment on a shared cluster. Run it first when both
nodes are free.

**B2. Communication vs computation decomposition.**
Set `profiling.enable_profiling=True` at each scale from B1, then classify kernels in
the emitted traces into NCCL/comm vs compute and report **comm fraction of step
time** at 1/2/4 GPUs. This is the attribution evidence: it converts "it scaled badly"
into "it scaled badly *because* comm went from 17% to ~95% of the step."

**B3. Interconnect characterisation + analytical model.**
Standalone microbenchmark (`torch.distributed.all_reduce` on tensors of
1 MB…1 GB, timed): measure achieved all-reduce bus bandwidth intra-node (2 GPU) and
inter-node (4 GPU). Then build the roofline:

```
predicted_comm_per_step = 2(N-1)/N × grad_bytes / measured_bus_bw
```

and compare predicted vs measured step time from B1/B2. **Agreement within ~20% is
the single most impressive artifact of this whole study** — it demonstrates you can
predict system behaviour, not just observe it.

### Tier 2 — The optimisation narrative

**B4. Gradient accumulation as communication reduction.** ga ∈ {1, 2, 4, 8} at
2 and 4 GPUs, compared **at equal tokens seen** (scale step count inversely with ga).
Report tokens/sec + comm fraction. On 1 GbE the 4-GPU effect should be large.

**B4b. (needs sign-off)** Add `no_sync` around the microbatch loop per the finding in
§2, then re-run B4. Expected: ga=8 at 4 GPUs approaches 8× fewer all-reduces per
token. Report before/after. This is the strongest single result available here.

**B5. Parallelism strategy at fixed scale (4 GPUs): DDP vs FSDP vs HSDP(2×2).**
Report tokens/sec, **peak memory/GPU**, and the largest model each strategy fits in
40GB. Requires only new bench configs at our model size — the existing `ddp`/`fsdp`/
`hsdp` presets assume a 16-GPU cluster (`dp_shard=16`) and will not run on 4. Zero
framework code changes. FSDP shards optimizer state, so the memory delta should be
large and directly supports a "enabled N× larger model on the same hardware" claim.

**B6. Activation-checkpointing tradeoff.** `activation_checkpoint.mode` ∈
{`none`, `selective`, `full`} at fixed scale. Report throughput vs peak memory.
Current runs use `selective` at only 24.6/40 GiB — `none` is likely a free
throughput win, and the resulting curve is a clean memory/compute tradeoff.

### Tier 3 — Model-architecture ablations (partly done; re-run rigorously)

**B7. MoE:** dense vs top_k ∈ {1,2,4} — re-run with equal-token comparison, and add
active-vs-total parameter accounting and memory alongside throughput.
**B8. Model size:** 163M vs ~976M — MFU and memory vs size; ties into B5's
"largest trainable model per strategy" result.

### Tier 4 — Still requires sign-off (unchanged from `README.md`)

Bucketing on/off + `bucket_cap_mb` sweep (hardcoded at 100 in `apply_ddp`; on 1 GbE
this could be a large effect), comm/compute overlap toggle, MLA vs standard MHA
(no uncompressed-KV path exists), YaRN perplexity (no eval harness exists).

---

## 4. Enabling work (do these first)

1. **Parameterise GPU count in `trainer.slurm`.** It hardcodes
   `--gpus-per-node=2` and `--nproc_per_node 2`; B1 needs 1, 2, and 4. Drive both
   from an env var (`GPUS_PER_NODE`, default 2) passed through `launch.sh`.
2. **Set NCCL env for the untested multi-node path.** No 2-node job has ever run
   here. Over plain Ethernet with several DOWN interfaces present, NCCL will likely
   hang or pick the wrong NIC without:
   ```
   export NCCL_SOCKET_IFNAME=eno5
   export GLOO_SOCKET_IFNAME=eno5
   export NCCL_IB_DISABLE=1
   ```
   `NCCL_DEBUG=INFO` is already set — confirm from the log that the ring actually
   crosses nodes on `eno5`.
3. **Try to restore `torch.compile`.** Inductor failed only because the venv's
   interpreter (`/usr/include/python3.12`) has no headers. A uv-managed or miniforge
   interpreter ships them: `uv python install 3.12 && uv sync -p <that>`, or
   `uv venv --python ~/miniforge3/bin/python3` (headers confirmed present at
   `~/miniforge3/include/python3.13/Python.h`). If it works, re-baseline — MFU
   should rise substantially and "raised MFU N×" becomes a legitimate bullet.
4. **Fix `summary.md`'s loss column** per §2(a) before anything is quoted.

---

## 5. Turning results into resume claims

Fill the blanks from measured data. Do not quote a number this study did not produce.

- *"Built a distributed LLM training framework from scratch (DeepSeek-V3 architecture:
  multi-head latent attention, mixture-of-experts, RoPE/YaRN) supporting DDP, FSDP,
  HSDP, tensor, pipeline, expert and context parallelism on a shared 4×A100 SLURM
  cluster."*
- *"Ran a strong-scaling study across 1/2/4 GPUs and measured scaling efficiency of
  __% intra-node vs __% across nodes; attributed the gap to communication via
  profiler trace analysis showing comm rising from __% to __% of step time."*
- *"Built an analytical roofline model of ring all-reduce cost from measured
  interconnect bandwidth that predicted observed step time within __%."*
- *"Identified that gradient accumulation was not suppressing per-microbatch
  gradient all-reduce; the fix cut communication volume __× and improved 4-GPU
  throughput by __%."* ← strongest bullet, contingent on B4b
- *"Compared DDP/FSDP/HSDP at fixed scale: FSDP reduced peak memory per GPU from
  __ GiB to __ GiB, enabling a __× larger model on 40GB A100s."*

**Interview framing.** Lead with the bottleneck analysis, not the model. The
honest, impressive story here is: *"I had a deliberately bad interconnect — PCIe with
no NVLink, 1 GbE with no RDMA — so I characterised it, predicted where scaling would
break, measured that it broke exactly there, and then recovered throughput by
attacking communication volume."* That demonstrates systems reasoning. "I trained a
1B model" does not — and would not survive the follow-up question, since
`bench_1b_2gpu` saw only ~0.4M tokens (50 steps × 8 × 1024) and is a memory/
throughput sanity check, **not** a trained model. Never describe it as one.

---

## 6. Rules for whoever runs this

- Check `squeue -p gpu-A100` before submitting; both nodes are shared and get taken.
- Never exceed 2 queued jobs (QOS). Prefer short step counts over long runs.
- 4-GPU runs hold both nodes — run them first when the cluster is free, and keep
  them ≤50 steps.
- Commit raw logs/CSVs and generated charts; every result must be reproducible from
  one scripted command.
- Do not modify reference-ported code (`nanotitan/**`) without explicit sign-off —
  see project rules in `spec.md`. Config files and `benchmarks/` are fair game.

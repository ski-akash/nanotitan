# Benchmarks

The reference repo (`torchfeather`) has no `benchmarks/` directory or conventions of
its own -- this suite is entirely ours, built to the spec's benchmarking scheme.

## Current scope: 2x A100

All seven configs below have now run for real on `csecluster` (`gpu-A100-01`/
`gpu-A100-02`) -- see `benchmarks/summary.md` for results.

### Configs (`nanotitan/config/default_configs.py`, `config_map`)

The six `bench_small_*` configs are DDP (`data_parallel_replicate_degree=2`,
`data_parallel_shard_degree=1`) on the `small` (163M-param) model, since
`apply_ddp`'s `bucket_cap_mb=100` gradient bucketing -- the thing most worth
measuring on PCIe/inter-node A100s -- only applies on that code path (FSDP buckets
differently). `bench_1b_2gpu` is the same DDP setup on a ~976M-param model, with a
smaller batch/seq_len/step-count, as a memory/throughput sanity check rather than a
full ablation.

| config | what it measures |
|---|---|
| `bench_small_2gpu` | baseline: dp=2, ga=1, MoE top_k=2 |
| `bench_small_ga4` | gradient accumulation ablation, ga=4 |
| `bench_small_ga8` | gradient accumulation ablation, ga=8 |
| `bench_small_dense` | MoE ablation: dense baseline (`n_dense_layers == n_layers`) |
| `bench_small_topk1` | MoE ablation: 1 active expert of 8 |
| `bench_small_topk4` | MoE ablation: 4 active experts of 8 |
| `bench_1b_2gpu` | ~976M-param model, same DDP setup, smaller batch/seq/steps |

### csecluster-specific fixes baked into these configs

None of these are architecture changes -- they're all working around real
environment constraints discovered while first running these jobs:

- **No compute-node internet access.** GPU nodes can't reach huggingface.co (only
  the login node can -- confirmed via an SSL cert mismatch), so the reference's
  live-streamed `fineweb` dataset can't load from a training job. Every
  `bench_*` config sets `training.dataset_path` to a local copy (one parquet shard +
  the HF repo's own README.md config, downloaded once via the login node to
  `assets/data/fineweb`, then referenced by absolute path since compute nodes may
  have a different cwd than expected).
- **No WandB credentials on csecluster.** `trainer.slurm` sets `WANDB_MODE=offline`
  so runs don't hard-fail without a `WANDB_API_KEY`. Use
  `collect_results_local.py` (below) instead of `collect_results.py` until a key is
  configured and the offline runs are `wandb sync`'d.
- **No `python3-dev` on compute nodes** (no `Python.h`, no root to install it), so
  `torch.compile`'s inductor backend can't build its C extensions. Every `bench_*`
  config sets `compile.enable = False`.
- **SLURM job-submit policy requires a `cd "$TMPDIR" || exit 1` line** in every job
  script (a site policy on csecluster, unrelated to this project) -- `trainer.slurm`
  has it near the top, followed immediately by `cd`-ing back to the project dir.

### Running

```
PARTITION=gpu-A100 ./benchmarks/run.sh              # submits all six bench_small_* configs
PARTITION=gpu-A100 ./benchmarks/run.sh bench_small_2gpu bench_small_ga4  # a subset
PARTITION=gpu-A100 ./benchmarks/run.sh bench_1b_2gpu # the 1B sanity check
```

Each submission is a `singlenode` (2-GPU) SLURM job via `launch.sh`/`trainer.slurm`,
same as any other named config. `WANDB_RUN_NAME` is set to the config name so results
are identifiable afterwards. Your account's SLURM QOS caps submissions at 2 jobs at
a time (`MaxSubmitPU=2`), so submit in batches of <=2.

### Collecting results

Metrics only ever go to WandB (`MetricsProcessor` has no local-file logging path).
With a configured `WANDB_API_KEY`, `collect_results.py` pulls them back via the
WandB API, averages the steady-state rows (skips the first `--skip-steps`, default
5, to drop warmup), and writes `benchmarks/raw/results.csv` (gitignored) +
`benchmarks/summary.md` (committed). Without one (the current state on
csecluster -- see above), use `collect_results_local.py` instead, which parses the
same metrics directly out of each config's `outputs/<config>/*.err` SLURM log --
same output files, no WandB needed:

```
python benchmarks/collect_results.py        # needs WANDB_API_KEY
python benchmarks/collect_results_local.py  # no WandB needed, run this on csecluster
python benchmarks/plot_results.py           # renders throughput.png, mfu.png, memory.png
```

## Open items -- not built yet, need a decision before touching ported code

The spec's benchmarking scheme also asks for these; each would require changing
code that was ported verbatim from the reference (or, for YaRN, adding a module the
reference doesn't have at all), so none are built without sign-off first:

- **Bucketing ON vs OFF.** `apply_ddp` (`nanotitan/distributed/model_parallel.py`)
  hardcodes `bucket_cap_mb=100` -- there's no config knob to disable/vary it. Doing
  this ablation needs a new `Parallelism`/`Comm` field threaded through to
  `apply_ddp`, which touches reference-derived code.
- **Comm/compute overlap ON vs OFF.** Same file -- overlap is implicit in
  `replicate()`/`fully_shard()`, not a separate toggle anywhere in the codebase.
- **Communication time / computation time ratio per step.** Not currently logged by
  `MetricsProcessor` at all. The lower-risk path is to reuse the profiling
  infrastructure that already exists (`tools/profiling.py`'s `maybe_enable_profiling`,
  Chrome trace dumps) rather than add new timing instrumentation to the training
  loop -- i.e. enable `profiling.enable_profiling` on a bench run and derive the
  ratio from the trace's kernel categories, post-hoc.
- **MLA vs standard multi-head attention.** The model only implements MLA -- K/V
  compression via `kv_lora_rank` is unconditional, there's no code path for
  uncompressed standard MHA to compare against. Reference has no such toggle either;
  adding one would be a genuine invention, not a port.
- **YaRN quality/perplexity across context lengths.** There's no eval/perplexity
  harness in this codebase yet -- `train.py` is training-only. This needs a new
  module the reference doesn't have.

## Also not built yet (lower priority)

- A real 1 vs 2 vs 4 GPU scaling-efficiency study (tokens/sec, scaling efficiency vs.
  ideal, per-GPU peak memory across GPU counts). Current focus per your direction is
  2-GPU benchmarking specifically; a 1-GPU baseline can reuse the existing `small`
  config directly (`torchrun --nproc_per_node=1`, no SLURM needed), and a 4-GPU point
  needs either a genuine 2-node SLURM allocation or an oversubscribed
  (`--nproc_per_node=4` on 2 physical GPUs) dry run for correctness-only, not real
  numbers -- see the conversation this suite came out of.

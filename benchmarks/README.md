# Benchmarks

The reference repo (`torchfeather`) has no `benchmarks/` directory or conventions of
its own -- this suite is entirely ours, built to the spec's benchmarking scheme.

## Current scope: 2x A100

Real runs need 2 free A100s on `csecluster`, which aren't reliably available yet (see
the cluster inventory in `spec.md`). Everything below is scaffolded so that as soon as
2 GPUs are free, it's a single `./benchmarks/run.sh` away from producing results --
nothing else needs to be built first for these six configs.

### Configs (`nanotitan/config/default_configs.py`, `config_map`)

All six are DDP (`data_parallel_replicate_degree=2`, `data_parallel_shard_degree=1`)
on the `small` model, since `apply_ddp`'s `bucket_cap_mb=100` gradient bucketing --
the thing most worth measuring on PCIe/inter-node A100s -- only applies on that code
path (FSDP buckets differently).

| config | what it measures |
|---|---|
| `bench_small_2gpu` | baseline: dp=2, ga=1, MoE top_k=2 |
| `bench_small_ga4` | gradient accumulation ablation, ga=4 |
| `bench_small_ga8` | gradient accumulation ablation, ga=8 |
| `bench_small_dense` | MoE ablation: dense baseline (`n_dense_layers == n_layers`) |
| `bench_small_topk1` | MoE ablation: 1 active expert of 8 |
| `bench_small_topk4` | MoE ablation: 4 active experts of 8 |

### Running

```
PARTITION=gpu-A100 ./benchmarks/run.sh              # submits all six
PARTITION=gpu-A100 ./benchmarks/run.sh bench_small_2gpu bench_small_ga4  # a subset
```

Each submission is a `singlenode` (2-GPU) SLURM job via `launch.sh`/`trainer.slurm`,
same as any other named config. `WANDB_RUN_NAME` is set to the config name so results
are identifiable afterwards.

### Collecting results

Metrics only ever go to WandB (`MetricsProcessor` has no local-file logging path) --
`collect_results.py` pulls them back via the WandB API, averages the steady-state
rows (skips the first `--skip-steps`, default 5, to drop warmup/compile), and writes:

- `benchmarks/raw/results.csv` -- raw per-config averages (gitignored)
- `benchmarks/summary.md` -- the same, as a committed markdown table

```
python benchmarks/collect_results.py
python benchmarks/plot_results.py   # renders throughput.png, mfu.png, memory.png
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

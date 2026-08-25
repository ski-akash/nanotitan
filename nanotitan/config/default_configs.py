from nanotitan.config.job_config import JobConfig
from nanotitan.model.model_args import DeepSeekV3ModelArgs
from nanotitan.model.moe.moe import MoEArgs


def get_deepseek_v3_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    )


def get_deepseek_v3_base_config() -> JobConfig:
    config = JobConfig()

    config.job.dump_folder = "./outputs"

    config.profiling.enable_profiling = False
    config.profiling.save_traces_folder = "profile_trace"
    config.profiling.profile_freq = 10
    config.profiling.enable_memory_snapshot = False
    config.profiling.save_memory_snapshot_folder = "memory_snapshot"

    config.metrics.log_freq = 1
    config.metrics.save_folder = "metrics"

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_deepseek_v3_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.2e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 50
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.local_batch_size = 5
    config.training.seq_len = 4096
    config.training.max_norm = 1.0
    config.training.steps = 500
    config.training.dataset = "fineweb"

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1

    config.checkpoint.enable = True
    config.checkpoint.folder = "checkpoint"
    config.checkpoint.interval = 100
    config.checkpoint.async_mode = "async"

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    config.compile.enable = True
    config.compile.components = ["model", "loss"]

    return config


def get_deepseek_v3_pp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 4

    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    return config


def get_deepseek_v3_hsdp_ep_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    return config


def get_deepseek_v3_fsdp_cp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 2
    return config


def get_deepseek_v3_fsdp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 8

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 8
    return config


def get_deepseek_v3_hsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    return config


def get_deepseek_v3_ddp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()

    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 16
    config.parallelism.data_parallel_shard_degree = 1
    return config


def get_deepseek_v3_fsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 16
    return config


def get_deepseek_v3_fsdp_ep_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 1
    return config


def get_deepseek_v3_fsdp_ep_etp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 2
    return config


def get_deepseek_v3_tiny_model_args() -> DeepSeekV3ModelArgs:
    # Small enough to iterate on quickly (even on CPU) while still exercising every
    # architecture feature: MLA, a dense first layer, and MoE layers after it.
    return DeepSeekV3ModelArgs(
        vocab_size=102400,  # must match the deepseek-moe-16b-base tokenizer's real vocab
        dim=128,
        inter_dim=384,
        moe_inter_dim=128,
        n_layers=4,
        n_dense_layers=1,
        n_heads=4,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=1,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=32,
    )


def get_deepseek_v3_tiny_config() -> JobConfig:
    """Fast-iteration config: fits comfortably on a single A100 with room to spare,
    for quickly checking that a change works end to end rather than for measuring
    anything."""
    config = JobConfig()

    config.job.dump_folder = "./outputs/tiny"

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_deepseek_v3_tiny_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 3e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 10
    config.lr_scheduler.decay_type = "cosine"

    config.training.local_batch_size = 4
    config.training.seq_len = 256
    config.training.max_norm = 1.0
    config.training.steps = 50
    config.training.dataset = "fineweb"

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1

    config.checkpoint.enable = False

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    config.compile.enable = True

    return config


def get_deepseek_v3_small_model_args() -> DeepSeekV3ModelArgs:
    # Big enough to be a meaningful benchmark target -- still runs unsharded on one
    # A100 (~170M params; well under 40/80GB even with AdamW's fp32 states), so it works
    # as the single-GPU baseline for the 1/2/4-GPU scaling study.
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=512,
        inter_dim=1536,
        moe_inter_dim=512,
        n_layers=8,
        n_dense_layers=1,
        n_heads=8,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=128,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        v_head_dim=64,
    )


def get_deepseek_v3_small_config() -> JobConfig:
    """Benchmark-run config: sized to meaningfully exercise MLA/MoE while staying
    well within an A100's 40/80GB even unsharded, so it can serve as the baseline for
    the 1/2/4-GPU scaling study and the bucketing/overlap/grad-accum ablations."""
    config = JobConfig()

    config.job.dump_folder = "./outputs/small"

    config.profiling.enable_profiling = False
    config.metrics.log_freq = 10

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_deepseek_v3_small_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 3e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 50
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.local_batch_size = 8
    config.training.seq_len = 2048
    config.training.max_norm = 1.0
    config.training.steps = 200
    config.training.dataset = "fineweb"

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1

    config.checkpoint.enable = False

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    config.compile.enable = True

    return config


def get_deepseek_v3_bench_small_2gpu_config() -> JobConfig:
    """Baseline for the 2x A100 benchmarking suite (benchmarks/): the "small" model,
    DDP-replicated across both GPUs (data_parallel_replicate_degree=2), which is the
    parallelism strategy apply_ddp's bucket_cap_mb=100 gradient bucketing actually
    applies to. global_batch_size is set explicitly so gradient_accumulation_steps
    works out to 1 here -- other bench_small_ga* configs vary only that."""
    config = get_deepseek_v3_small_config()

    config.job.dump_folder = "./outputs/bench_small_2gpu"

    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 1

    config.training.global_batch_size = (
        config.training.local_batch_size * 2
    )  # ga=1 at dp_degree=2

    # csecluster's GPU compute nodes have no internet access (only the login node
    # does), so the reference's live-streamed "fineweb" dataset can't be reached from
    # a training job. Points at a local copy (one parquet shard, mirroring the HF
    # repo's own data/<dump>/*.parquet layout + its README.md config, downloaded via
    # the login node) instead -- see benchmarks/README.md.
    config.training.dataset_path = (
        "/userhome/mtech/akashc1005/nanotitan/assets/data/fineweb"
    )

    # csecluster's GPU compute nodes have no python3-dev (no Python.h), which
    # torch.compile's inductor backend needs to build its C extensions, and there's
    # no root access to install it. Disabled for these bench runs until that's
    # resolved -- see benchmarks/README.md.
    config.compile.enable = False

    return config


def _get_bench_scale_base_config(dp_degree: int) -> JobConfig:
    """Shared base for the B1 strong-scaling study (benchmarks/BENCHMARK_PLAN.md).

    Strong scaling means fixed total work, split over more GPUs: global_batch_size,
    seq_len and step count are identical at every point, so all three configs see
    exactly the same 655,360 tokens and differ only in how many GPUs divide it.
    local_batch_size therefore shrinks as dp_degree grows, keeping
    gradient_accumulation_steps == 1 throughout (so this measures scaling, not
    accumulation -- that's B4).

    seq_len is 1024 rather than the 2048 used by bench_small_*, so the 1-GPU point
    (which carries the whole global batch on one device) lands at roughly the same
    activation memory as the already-verified 24.65 GiB 2-GPU run.
    """
    config = get_deepseek_v3_bench_small_2gpu_config()

    global_batch_size = 16
    assert global_batch_size % dp_degree == 0, (global_batch_size, dp_degree)

    config.parallelism.data_parallel_replicate_degree = dp_degree
    config.parallelism.data_parallel_shard_degree = 1

    config.training.seq_len = 1024
    config.training.local_batch_size = global_batch_size // dp_degree
    config.training.global_batch_size = global_batch_size  # => ga == 1 at every point
    config.training.steps = 40

    # One row per step: only 40 steps total, and the first 10 are discarded as
    # warmup, so per-step logging is what makes the remaining 30 a usable sample.
    config.metrics.log_freq = 1

    return config


def get_deepseek_v3_bench_scale_1gpu_config() -> JobConfig:
    """B1 strong-scaling, world_size=1. Launch: GPUS_PER_NODE=1 ./launch.sh singlenode"""
    config = _get_bench_scale_base_config(dp_degree=1)
    config.job.dump_folder = "./outputs/bench_scale_1gpu"
    return config


def get_deepseek_v3_bench_scale_2gpu_config() -> JobConfig:
    """B1 strong-scaling, world_size=2, intra-node over PCIe/UPI (`SYS`, no NVLink).
    Launch: ./launch.sh singlenode"""
    config = _get_bench_scale_base_config(dp_degree=2)
    config.job.dump_folder = "./outputs/bench_scale_2gpu"
    return config


def get_deepseek_v3_bench_scale_4gpu_config() -> JobConfig:
    """B1 strong-scaling, world_size=4, spanning 2 nodes -- so gradient all-reduce
    crosses the 1GbE inter-node link (no InfiniBand/RDMA). This is the point the
    whole study exists to measure. Launch: ./launch.sh multinode"""
    config = _get_bench_scale_base_config(dp_degree=4)
    config.job.dump_folder = "./outputs/bench_scale_4gpu"
    return config


def get_deepseek_v3_bench_1b_model_args() -> DeepSeekV3ModelArgs:
    # ~976M total params (dense 387M + all-experts sparse 589M; ~584M active per
    # forward pass with top_k=2 of 8 experts). At fp32, weights+grads+AdamW states
    # alone are ~15.6GB -- leaves headroom on a 40GB A100 for activations, but not
    # as much margin as the 163M "small" model, hence the smaller batch/seq_len
    # below versus bench_small_2gpu.
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=1408,
        inter_dim=3840,
        moe_inter_dim=1408,
        n_layers=12,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=88,
        qk_rope_head_dim=32,
        v_head_dim=128,
    )


def get_deepseek_v3_bench_1b_2gpu_config() -> JobConfig:
    """~1B-param counterpart to bench_small_2gpu -- same DDP-on-2-GPUs setup, local
    dataset, and compile disabled (all for the same csecluster reasons), but a
    smaller batch_size/seq_len than the 163M model since there's less memory
    headroom left for activations. Fewer steps too (50, not 200) since this is a
    memory/throughput sanity check on a bigger model, not a full ablation."""
    config = get_deepseek_v3_bench_small_2gpu_config()

    config.job.dump_folder = "./outputs/bench_1b_2gpu"

    config.model.args = get_deepseek_v3_bench_1b_model_args()

    config.training.local_batch_size = 4
    config.training.seq_len = 1024
    config.training.steps = 50
    config.training.global_batch_size = config.training.local_batch_size * 2  # ga=1

    return config


def get_deepseek_v3_bench_small_ga4_config() -> JobConfig:
    """Gradient-accumulation ablation, ga=4, at the same dp_degree=2 as
    bench_small_2gpu."""
    config = get_deepseek_v3_bench_small_2gpu_config()
    config.job.dump_folder = "./outputs/bench_small_ga4"
    config.training.global_batch_size = config.training.local_batch_size * 2 * 4
    return config


def get_deepseek_v3_bench_small_ga8_config() -> JobConfig:
    """Gradient-accumulation ablation, ga=8, at the same dp_degree=2 as
    bench_small_2gpu."""
    config = get_deepseek_v3_bench_small_2gpu_config()
    config.job.dump_folder = "./outputs/bench_small_ga8"
    config.training.global_batch_size = config.training.local_batch_size * 2 * 8
    return config


def get_deepseek_v3_bench_small_dense_config() -> JobConfig:
    """MoE ablation: dense baseline -- every layer uses the shared FeedForward
    instead of the MoE layer (n_dense_layers == n_layers), for comparison against
    bench_small_2gpu's MoE (top_k=2 of 8 experts)."""
    config = get_deepseek_v3_bench_small_2gpu_config()
    config.job.dump_folder = "./outputs/bench_small_dense"
    config.model.args.n_dense_layers = config.model.args.n_layers
    return config


def get_deepseek_v3_bench_small_topk1_config() -> JobConfig:
    """MoE ablation: fewer active experts per token (top_k=1 of 8) than
    bench_small_2gpu's top_k=2, to compare active-vs-total-param tradeoffs."""
    config = get_deepseek_v3_bench_small_2gpu_config()
    config.job.dump_folder = "./outputs/bench_small_topk1"
    config.model.args.moe_args.top_k = 1
    return config


def get_deepseek_v3_bench_small_topk4_config() -> JobConfig:
    """MoE ablation: more active experts per token (top_k=4 of 8) than
    bench_small_2gpu's top_k=2, to compare active-vs-total-param tradeoffs."""
    config = get_deepseek_v3_bench_small_2gpu_config()
    config.job.dump_folder = "./outputs/bench_small_topk4"
    config.model.args.moe_args.top_k = 4
    return config


config_map = {
    "tiny": get_deepseek_v3_tiny_config,
    "small": get_deepseek_v3_small_config,
    "bench_small_2gpu": get_deepseek_v3_bench_small_2gpu_config,
    "bench_1b_2gpu": get_deepseek_v3_bench_1b_2gpu_config,
    "bench_scale_1gpu": get_deepseek_v3_bench_scale_1gpu_config,
    "bench_scale_2gpu": get_deepseek_v3_bench_scale_2gpu_config,
    "bench_scale_4gpu": get_deepseek_v3_bench_scale_4gpu_config,
    "bench_small_ga4": get_deepseek_v3_bench_small_ga4_config,
    "bench_small_ga8": get_deepseek_v3_bench_small_ga8_config,
    "bench_small_dense": get_deepseek_v3_bench_small_dense_config,
    "bench_small_topk1": get_deepseek_v3_bench_small_topk1_config,
    "bench_small_topk4": get_deepseek_v3_bench_small_topk4_config,
    "hsdp": get_deepseek_v3_hsdp_config,
    "ddp": get_deepseek_v3_ddp_config,
    "fsdp": get_deepseek_v3_fsdp_config,
    "pp_tp": get_deepseek_v3_pp_tp_config,
    "fsdp_tp": get_deepseek_v3_fsdp_tp_config,
    "fsdp_cp": get_deepseek_v3_fsdp_cp_config,
    "hsdp_ep": get_deepseek_v3_hsdp_ep_config,
    "fsdp_ep_tp": get_deepseek_v3_fsdp_ep_tp_config,
    "fsdp_ep_etp": get_deepseek_v3_fsdp_ep_etp_config,
}


def get_config(name: str) -> JobConfig:
    return config_map[name]()

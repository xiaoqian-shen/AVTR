"""Environment and kernel setup for the in-process vLLM engine the Qwen3.5 adapters run.

Kept free of torch/vllm imports at module level so tools outside the harness can depend on it
without pulling in vLLM.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from typing import Any

from loguru import logger


def prepare_vllm_env() -> None:
    """Run vLLM v1's EngineCore in this process. REQUIRED, not a tuning knob: ``pin_step_order``
    and the imagine chain (``imagine/chain.py``) reach into the in-process EngineCore (its batch
    queue, scheduler and executor), which a separate engine process would hide. Must happen
    before vllm is imported."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Harmless without tensor parallelism, where no all-reduce runs: they only keep vLLM/torch off the
    # symmetric-memory (multicast) all-reduce path, which some drivers and containers lack.
    os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")


def _temp_dir_removed_at_exit(prefix: str, parent: str | None = None) -> str:
    """A new private directory, deleted with its contents when this process exits."""
    path = tempfile.mkdtemp(prefix=prefix, dir=parent)
    owner = os.getpid()

    def remove() -> None:
        # A forked child inherits atexit hooks; only the creating process may delete the dir.
        if os.getpid() == owner:
            shutil.rmtree(path, ignore_errors=True)

    atexit.register(remove)
    return path


def isolate_compile_caches(rank: int) -> None:
    """Give inductor/triton/vLLM a private, writable cache dir per rank.

    With enforce_eager=False vLLM turns on torch.compile, and inductor defaults to a
    shared ``/var/tmp/torchinductor_<user>``. With several ranks on one host all of them land on
    that same path and race: a rank looks up a cubin another rank is still writing (or that
    got cleaned) and dies with ``Cubin file saved by TritonBundler not found``. Invisible on
    a single-GPU run because there is exactly one writer and nothing prunes the dir.

    The dirs go under ``rank<r>/`` of a fresh ``avtr_compile_*`` dir in ``$AVTR_COMPILE_CACHE_ROOT``
    (default: the system temp dir), deleted when the process exits, so a sweep of runs leaves no
    caches behind. A cache variable that is already set is left alone.
    """
    caches = (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("VLLM_CACHE_ROOT", "vllm"),
    )
    if all(var in os.environ for var, _ in caches):
        return
    parent = os.environ.get("AVTR_COMPILE_CACHE_ROOT") or None
    if parent:
        os.makedirs(parent, exist_ok=True)
    root = _temp_dir_removed_at_exit("avtr_compile_", parent)
    for var, stem in caches:
        os.environ.setdefault(var, os.path.join(root, f"rank{rank}", stem))


def ensure_process_group(rank: int, world_size: int) -> None:
    """``external_launcher`` reads RANK/WORLD_SIZE and expects an initialized process
    group; without one vLLM dies with ``KeyError: 'RANK'``.

    Under accelerate/torchrun the group already exists and the env is set, so this is a
    no-op. Standalone (a single-process ``python -m lmms_eval`` run) it creates a file-backed
    single-process gloo group, which needs no free TCP port; the store file is deleted when the
    process exits.
    """
    import torch.distributed as dist

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    if dist.is_available() and not dist.is_initialized():
        store = os.path.join(_temp_dir_removed_at_exit("avtr_pg_"), "store")
        dist.init_process_group(
            backend="gloo", init_method=f"file://{store}", rank=0, world_size=1
        )


# The config each fla autotuner chooses for Qwen3.5-9B's GatedDeltaNet on H100: the most common pick
# over many processes, whose runners-up timed within noise of it. With one config left, Triton
# neither compiles nor times the others at the first prefill -- otherwise a large share of the wall
# time of a small image task -- and every rank of every run gets the same kernels.
# Kernel name -> (meta-parameters, num_warps, num_stages).
FLA_PINNED_CONFIGS: dict[str, tuple[dict[str, int], int, int]] = {
    "chunk_fwd_kernel_o": ({"BK": 64, "BV": 64}, 2, 2),
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ({"BV": 64}, 4, 2),
    "chunk_local_cumsum_scalar_kernel": ({}, 2, 3),
    "chunk_scaled_dot_kkt_fwd_kernel": ({"BK": 64}, 2, 3),
    "merge_16x16_to_64x64_inverse_kernel": ({}, 2, 3),
    "recompute_w_u_fwd_kernel": ({}, 2, 2),
}


def pinned_config(name: str, configs: list[Any]) -> Any | None:
    """The entry of ``configs`` that ``FLA_PINNED_CONFIGS`` names for kernel ``name``, or None when
    the kernel is not pinned or this vllm build no longer offers that config."""
    want = FLA_PINNED_CONFIGS.get(name)
    if want is None:
        return None
    kwargs, num_warps, num_stages = want
    for config in configs:
        if (config.kwargs, config.num_warps, config.num_stages) == want:
            return config
    logger.warning(
        "fla pin: {} offers no config {} with num_warps={} num_stages={}; it keeps autotuning",
        name,
        kwargs,
        num_warps,
        num_stages,
    )
    return None


def _try_import(importlib: Any, module: str) -> Any | None:
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001 -- the module may not exist in this vllm build
        return None


def patch_gdn_kernels(pin_configs: bool = True) -> None:
    """Route Qwen3.5's GatedDeltaNet off the FlashInfer sm90 kernel, cap the fla forward
    tiles and, with ``pin_configs``, pin each fla autotuner to ``FLA_PINNED_CONFIGS``.

    The first keeps every GPU type on the same Triton/FLA GatedDeltaNet kernels (vLLM would
    otherwise JIT-compile FlashInfer's kernel on sm90). The second is
    load-bearing: vLLM's fla autotune picks an oversized BV tile that needs more shared memory
    than the device limit and faults with ``cudaErrorLaunchFailure`` on the first long
    prefill. Best-effort and idempotent; a no-op if this vllm build lays the modules out
    differently. Must run before the first kernel launch, i.e. before ``LLM()``.
    """
    import importlib

    # Older vLLM releases define ChunkGatedDeltaRule in the Qwen3-Next model file; newer ones moved
    # it to the shared GatedDeltaNet layer.
    for module in (
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "vllm.model_executor.models.qwen3_next",
    ):
        rule = getattr(_try_import(importlib, module), "ChunkGatedDeltaRule", None)
        if rule is not None:
            rule.forward_cuda = rule.forward_native
            break
    else:
        logger.warning("GDN forward_native patch skipped: no ChunkGatedDeltaRule in this vllm")

    # Counted and logged like the training-side twin (the fla autotune tile cap of the avtr
    # training code). Without the count there is no way to tell a cap that worked
    # from one that silently found nothing -- which matters, because a qvhighlights eval died in
    # the Triton autotuner benchmarking chunk_scaled_dot_kkt on a long video prefill
    # with this cap supposedly already applied.
    pruned = pinned = 0
    base = "vllm.model_executor.layers.fla.ops"
    for sub in (
        "chunk_o",
        "chunk_scaled_dot_kkt",
        "chunk_delta_h",
        "wy_fast",
        "solve_tril",
        "cumsum",
    ):
        try:
            mod = importlib.import_module(f"{base}.{sub}")
        except Exception:  # noqa: BLE001 -- optional module
            continue
        for name in dir(mod):
            tuner: Any = getattr(mod, name, None)
            for _ in range(4):  # descend heuristics/autotune wrappers to the Autotuner
                cfgs = getattr(tuner, "configs", None)
                if isinstance(cfgs, list) and cfgs and hasattr(cfgs[0], "kwargs"):
                    kept = [
                        c
                        for c in cfgs
                        if all(
                            v <= 64 for k, v in c.kwargs.items() if k in ("BK", "BV")
                        )
                    ]
                    if not kept:
                        # Never fall back to the full list: that puts every oversized tile back
                        # and is indistinguishable from a successful cap. An oversized tile needs
                        # more shared memory than the device limit and faults on the first long
                        # prefill.
                        logger.warning(
                            "fla tile cap: {}.{} has NO config with BK/BV<=64; leaving "
                            "{} oversized config(s) in place, long prefills may fault",
                            sub,
                            name,
                            len(cfgs),
                        )
                    else:
                        pin = pinned_config(name, kept) if pin_configs else None
                        if pin is not None:
                            kept = [pin]
                            pinned += 1
                        if len(kept) != len(cfgs):
                            setattr(tuner, "configs", kept)  # noqa: B010
                            pruned += 1
                    break
                tuner = getattr(tuner, "fn", None)
    logger.info(
        "fla autotune: pruned {} config list(s), pinned {} of {} kernel(s)",
        pruned,
        pinned,
        len(FLA_PINNED_CONFIGS),
    )


class _UnfinishedFuture:
    """A step's output future that reports itself unfinished until it is read."""

    def __init__(self, future: Any) -> None:
        self._future = future

    def done(self) -> bool:
        return False

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout)


def pin_step_order(llm: Any) -> bool:
    """Make the batch a request runs in independent of how fast the GPU was.

    Under async scheduling (vLLM's default for ``external_launcher``),
    ``EngineCore.step_with_batch_queue`` queues a step into an empty queue and returns nothing while
    the GPU runs it -- unless the GPU already finished by the time it checks
    (``batch_queue[-1][0].done()``), in which case it returns the output at once. The imagine chain
    submits a row's next request from that output, so the branch decides which batch the request
    joins, and vLLM's numerics depend on the batch: one rank of an erqa run that took the early
    branch scored most of its rows differently from a rerun of the same build. Reporting the step
    unfinished always takes the pipelined branch. Returns False when there is no batch queue to pin.
    """
    core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    if core is None or getattr(core, "batch_queue", None) is None:
        return False
    executor = core.model_executor
    sample_tokens = executor.sample_tokens

    def pinned(grammar_output: Any, non_block: bool = False) -> Any:
        out = sample_tokens(grammar_output, non_block=non_block)
        return _UnfinishedFuture(out) if non_block else out

    executor.sample_tokens = pinned
    return True


def deterministic_compilation_config() -> dict[str, Any]:
    """vLLM ``compilation_config`` that makes the compiled kernels a function of the code alone.

    vLLM passes ``inductor_compile_config`` to Inductor as config patches. Deterministic mode picks
    reduction configs without timing them; it forbids the benchmark behind vLLM's default
    ``benchmark_combo_kernel=True``, so combo kernels stay on but are chosen by heuristic. The two
    remote caches are off because they would hand back kernels another job tuned.
    """
    return {
        "inductor_compile_config": {
            "deterministic": True,
            "fx_graph_remote_cache": False,
            "autotune_remote_cache": False,
            "combo_kernels": True,
            "benchmark_combo_kernel": False,
        }
    }

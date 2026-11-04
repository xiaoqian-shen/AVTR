"""Runtime FLA tile-size limits for Qwen3.5 training and evaluation."""

import logging
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)

_applied_patches: set[str] = set()


def apply_fla_bwd_patch() -> None:
    """Cap CONST_TILING in fla's chunk_bwd_dv_local for H100 stability.

    fla uses a larger CONST_TILING on Hopper, which crashes the Triton
    autotuner with oversized BV tiles for Qwen3.5/3.6 GDN linear-attn
    via CUDA illegal memory access. The cap makes the V loop run more
    iterations; correctness preserved.

    No-op if fla is not installed.
    """
    if "fla_bwd" in _applied_patches:
        return
    _applied_patches.add("fla_bwd")

    try:
        import triton
        from fla.ops.common import chunk_o as chunk_o_module
        from fla.ops.utils import prepare_chunk_indices
        from fla.utils import check_shared_mem
    except ImportError:
        logger.debug("fla not available, skipping bwd patch")
        return

    def _patched_chunk_bwd_dv_local(
        q: torch.Tensor,
        k: torch.Tensor,
        do: torch.Tensor,
        g: torch.Tensor | None = None,
        g_gamma: torch.Tensor | None = None,
        A: torch.Tensor | None = None,
        scale: float = 1.0,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        B, T, H, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
        V = do.shape[-1]
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        if check_shared_mem("hopper", k.device.index):
            CONST_TILING = 64
        elif check_shared_mem():
            CONST_TILING = 64
        else:
            CONST_TILING = 32
        BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
        BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
        if cu_seqlens is None:
            NT = triton.cdiv(T, BT)
        else:
            assert chunk_indices is not None
            NT = len(chunk_indices)

        dv = torch.empty_like(do)
        grid = (NT, B * H)
        chunk_o_module.chunk_bwd_kernel_dv_local[grid](
            q=q,
            k=k,
            g=g,
            g_gamma=g_gamma,
            A=A,
            do=do,
            dv=dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        )
        return dv

    chunk_o_module.chunk_bwd_dv_local = _patched_chunk_bwd_dv_local
    try:
        from fla.ops.gated_delta_rule import chunk as gdr_chunk

        gdr_chunk.chunk_bwd_dv_local = _patched_chunk_bwd_dv_local
    except (ImportError, AttributeError):
        pass

    logger.info("fla bwd patch applied: capped CONST_TILING=64 for chunk_bwd_dv_local")


# BK/BV above this overflow Hopper shared memory at GDN's head dim (see apply_fla_bwd_patch).
_FLA_MAX_TILE: int = 64
# fla keeps these under ops/common/ (solve_tril under ops/utils/); the vLLM-vendored copy flattens
# them under ops/. Probed in that order rather than pinning one layout.
_FLA_TILED_MODULES: tuple[str, ...] = (
    "chunk_o",
    "chunk_scaled_dot_kkt",
    "chunk_delta_h",
    "chunk_h",
    "wy_fast",
    "solve_tril",
)
_FLA_MODULE_ROOTS: tuple[str, ...] = ("fla.ops.common", "fla.ops.utils", "fla.ops")


def apply_fla_autotune_tile_cap() -> None:
    """Drop BK/BV configs above `_FLA_MAX_TILE` from EVERY fla autotuner, not just `chunk_bwd_dv_local`.

    `apply_fla_bwd_patch` swaps one function, so it only fixes the tile choice inside
    `chunk_bwd_dv_local`. The identical oversized-BV overflow faults the other autotuned GDN backward
    kernels too -- observed as `chunk_bwd_kernel_dqkwg` (same `fla/ops/common/chunk_o.py`) raising
    a CUDA illegal memory access from inside the autotuner's own `do_bench`. Pruning the oversized
    candidates covers the whole family at once; it is what the vLLM eval adapter already does to
    its vendored copy.

    MUST run before the first GDN backward. Sequence lengths keep varying from step to step, so every
    new length is a fresh autotune key, and each fresh autotune is another chance to benchmark the
    faulting tile.

    Best-effort and idempotent: a module this fla build lays out differently is skipped, and a prune
    that would empty a candidate list is discarded instead.
    """
    if "fla_autotune_tile_cap" in _applied_patches:
        return
    _applied_patches.add("fla_autotune_tile_cap")

    import importlib

    pruned = 0
    for name in _FLA_TILED_MODULES:
        mod: Any = None
        for root in _FLA_MODULE_ROOTS:
            try:
                mod = importlib.import_module(f"{root}.{name}")
                break
            except Exception:  # noqa: BLE001 -- optional module in this fla layout
                continue
        if mod is None:
            continue
        for attr in dir(mod):
            tuner: Any = getattr(mod, attr, None)
            for _ in range(4):  # descend heuristics/autotune wrappers to the Autotuner
                if tuner is None:  # ran off the end of the wrapper chain
                    break
                configs = getattr(tuner, "configs", None)
                if (
                    isinstance(configs, list)
                    and configs
                    and hasattr(configs[0], "kwargs")
                ):
                    kept = [
                        c
                        for c in configs
                        if all(
                            v <= _FLA_MAX_TILE
                            for k, v in c.kwargs.items()
                            if k in ("BK", "BV")
                        )
                    ]
                    if kept and len(kept) != len(configs):
                        tuner.configs = kept
                        pruned += 1
                    break
                tuner = getattr(tuner, "fn", None)
    logger.info("fla autotune tile cap: pruned %d autotuner config list(s)", pruned)

"""Qwen3.5 evaluation with one vLLM engine per GPU.

Inherits prompt construction and answer handling from the Transformers adapter.
Data-parallel requests use an in-process engine with deterministic kernel setup.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.chunk_dedup import dedup_slots
from lmms_eval.models.chunk_prefetch import prefetch, split_chunks
from lmms_eval.models.model_utils.qwen.vision_process import (
    fetch_image,
    fetch_video,
)
from lmms_eval.models.model_utils.qwen3_5_gen import (
    has_reasoning,
    run_per_gen_kwargs,
)
from lmms_eval.models.model_utils.vllm_setup import (
    deterministic_compilation_config,
    ensure_process_group,
    isolate_compile_caches,
    patch_gdn_kernels,
    pin_step_order,
    prepare_vllm_env,
)
from lmms_eval.models.qwen3_5 import Qwen3_5
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


# Rows in the first chunk of a stream (see Qwen3_5VLLM._iter_chunks): one round of the default
# number of decode workers.
STREAM_FIRST_CHUNK = 8


def _fetch_visual(ele: dict) -> Any:
    """One image or video, as ``process_vision_info(..., return_video_metadata=True)`` returns it."""
    if "image" in ele or "image_url" in ele:
        return fetch_image(ele, image_patch_size=16)
    return fetch_video(ele, image_patch_size=16, return_video_metadata=True)


@register_model("qwen3_5_vllm")
class Qwen3_5VLLM(Qwen3_5):
    """Qwen3.5 through vLLM, with Qwen3_5's prompt/answer handling."""

    # Qwen3_5.__init__ never runs here, so these class values are the generation options
    # this backend runs with: the eval semantics it was validated with.
    _floor_max_new_tokens = True
    _extract_final_answer = True
    min_new_tokens = 0
    sample_temperature = None
    sc_k = 1

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        batch_size: int | str = 1,  # accepted and ignored: vLLM batches continuously
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 16384 * 28 * 28,
        video_min_pixels: int = 128 * 28 * 28,
        video_max_pixels: int = 768 * 28 * 28,
        video_total_pixels: int = 115200 * 28 * 28,
        min_frames: int = 4,
        max_frames: int = 768,
        nframes: int | None = None,
        fps: float | None = 2.0,
        interleave_visuals: bool = False,
        enable_thinking: bool = False,
        bare_think: bool = False,
        raw_dump_path: str | None = None,
        max_model_len: int = 40960,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 32,
        # Also determines the multimodal encoder cache; it must fit the largest media input.
        max_num_batched_tokens: int = 32768,
        # Per-request prefill cap, independent of the encoder-cache budget. Zero uses vLLM's default.
        long_prefill_token_threshold: int = 0,
        # Order the queue by the per-request `priority` instead of arrival. The imagine adapter
        # needs it: a row's follow-up requests outrank rows that have not started, so each is
        # scheduled while the row's prefix is still cached.
        scheduling_priority: bool = False,
        # Disable CUDA graphs when eager execution is requested.
        enforce_eager: bool = False,
        # Use deterministic compilation and fixed GatedDeltaNet kernel configurations.
        deterministic_compile: bool = True,
        # Maximum images per prompt; this limit does not preallocate their memory.
        limit_images: int = 32,
        # Big enough that vLLM always has far more than max_num_seqs queued (chunking
        # too finely throttles continuous batching), small
        # enough that media decode for the next chunk overlaps this one's generation.
        chunk_size: int = 128,
        vision_workers: int = 8,
        # CPU threads for processor rendering; zero preserves the current setting.
        render_threads: int = 8,
        # None keeps vLLM's default processor cache. The imagine adapter disables it: it renders a
        # row once and resubmits the rendered input, which desyncs the default cache's
        # sender/receiver pair (vllm/multimodal/cache.py).
        mm_processor_cache_gb: float | None = None,
        **kwargs: Any,
    ) -> None:
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        # Deliberately skip Qwen3_5.__init__: it loads the full HF weights this backend
        # does not use, so the grandparent's __init__ is called directly.
        lmms.__init__(self)

        from accelerate import Accelerator

        acc = Accelerator()
        self._rank = acc.process_index
        self._world_size = acc.num_processes
        if self._world_size > 1:
            self.accelerator = acc
        # The inherited `device` property reads _device, and evaluator.evaluate() touches
        # it only when gathering across ranks -- so omitting it passed every single-GPU
        # run here and blew up with AttributeError on the first multi-rank job.
        self._device = torch.device(f"cuda:{acc.local_process_index}")
        self.device_map = f"cuda:{acc.local_process_index}"

        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.nframes = nframes
        self.fps = fps
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = enable_thinking
        self._bare_think = bool(bare_think)
        self._raw_dump_path = raw_dump_path
        self._chunk_size = max(1, int(chunk_size))
        self._vision_workers = max(1, int(vision_workers))
        self.batch_size_per_gpu = 1
        self.use_cache = True

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer.padding_side = "left"
        self._config = AutoConfig.from_pretrained(pretrained)

        prepare_vllm_env()
        isolate_compile_caches(self._rank)
        ensure_process_group(self._rank, self._world_size)
        patch_gdn_kernels(pin_configs=deterministic_compile)
        from vllm import LLM

        self._llm = LLM(
            model=pretrained,
            tokenizer=pretrained,
            tensor_parallel_size=1,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # Keep each GDN forward over a short chunk rather than one giant scan.
            max_num_batched_tokens=max_num_batched_tokens,
            **({"scheduling_policy": "priority"} if scheduling_priority else {}),
            **(
                {"long_prefill_token_threshold": long_prefill_token_threshold}
                if long_prefill_token_threshold
                else {}
            ),
            max_num_seqs=max_num_seqs,
            # Proven-loadable for this hybrid; turning it off crashed engine init.
            enable_prefix_caching=True,
            enforce_eager=enforce_eager,
            **(
                {"compilation_config": deterministic_compilation_config()}
                if deterministic_compile and not enforce_eager
                else {}
            ),
            trust_remote_code=True,
            limit_mm_per_prompt={"image": limit_images, "video": 1},
            **(
                {"mm_processor_cache_gb": float(mm_processor_cache_gb)}
                if mm_processor_cache_gb is not None
                else {}
            ),
            seed=0,
        )
        pinned = pin_step_order(self._llm)
        sched = self._llm.llm_engine.vllm_config.scheduler_config
        eval_logger.info(
            "qwen3_5_vllm rank {}/{}: engine up (max_model_len={}, max_num_seqs={}, "
            "max_num_batched_tokens={}, encoder_cache_size={}, step order pinned={})",
            self._rank,
            self._world_size,
            max_model_len,
            max_num_seqs,
            getattr(sched, "max_num_batched_tokens", "?"),
            getattr(sched, "encoder_cache_size", "?"),
            pinned,
        )
        self._warm_up()
        if render_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(render_threads)

    def _warm_up(self) -> None:
        """Run a few throwaway prefills so the first real engine step does not build kernels.

        The GatedDeltaNet Triton kernels compile on first use, once per specialization of their
        integer arguments (e.g. whether a token count is aligned, one sequence or many), and the
        vision tower pays its own first-call setup. That stalled the first real step for a long
        time with every queued request behind it, and the imagine stream admits only its first chunk
        before that step.
        """
        from PIL import Image
        from vllm import SamplingParams

        t0 = time.perf_counter()
        image_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        image = Image.new("RGB", (448, 448), (127, 127, 127))
        # Distinct ids per prompt, so no prompt is a prefix-cache hit on another.
        text = [
            {"prompt_token_ids": list(range(base, base + n))}
            for base, n in ((1000, 1000), (3000, 517), (5000, 1008), (7000, 64))
        ]
        sp = SamplingParams(max_tokens=2, temperature=0.0)
        for batch in (
            [{"prompt": image_prompt, "multi_modal_data": {"image": [image]}}]
            + text[:2],
            text[2:],
        ):
            self._llm.generate(batch, sp, use_tqdm=False)
        eval_logger.info(
            "qwen3_5_vllm rank {}: warm-up {:.1f}s",
            self._rank,
            time.perf_counter() - t0,
        )

    @property
    def model(self):
        return self._llm

    def _mm_rows(
        self, batched_messages: list, texts: list[str], pool: Executor
    ) -> list[dict]:
        """One vLLM input row per sample: prompt text plus its own vision payload. Each image and
        video is read and resized on ``pool``, as ``process_vision_info`` would."""
        visuals = [
            [
                ele
                for m in msg
                if isinstance(m.get("content"), list)
                for ele in m["content"]
                if "image" in ele or "image_url" in ele or "video" in ele
            ]
            for msg in batched_messages
        ]
        flat = [ele for row_visuals in visuals for ele in row_visuals]
        fetched = iter(list(pool.map(_fetch_visual, flat)))
        rows: list[dict] = []
        for row_visuals, prompt in zip(visuals, texts):
            imgs, vids = [], []
            for ele in row_visuals:
                media = next(fetched)
                (imgs if "image" in ele or "image_url" in ele else vids).append(media)
            row: dict[str, Any] = {"prompt": prompt}
            mm: dict[str, Any] = {}
            if imgs:
                mm["image"] = imgs
            if vids:
                mm["video"] = vids
            if mm:
                row["multi_modal_data"] = mm
            if vids:
                # process_vision_info's video kwargs: the frames are sampled and resized already.
                row["mm_processor_kwargs"] = {
                    "do_sample_frames": False,
                    "do_resize": False,
                }
            rows.append(row)
        return rows

    def _prepare_chunk(self, chunk):
        """Decode distinct documents once and construct each request's multimodal input.

        Task, split, and visual callback are resolved separately for every request.
        """
        contexts = [c for c, _, _, _, _ in chunk]
        ids = [d for _, d, _, _, _ in chunk]
        docs = [self.task_dict[tk][sp][d] for _, d, tk, sp, _ in chunk]
        fns = [f for _, _, _, _, f in chunk]
        # Decode each DISTINCT doc once, not each request. A task with `repeats: K` is expanded by
        # evaluator.py into K identical requests (`cloned_reqs.extend([req] * req.repeats)`), so a
        # repeated task decoded every clip K times. Decode is pure CPU and already the pipeline's
        # bottleneck, so the waste showed up as an idle GPU.
        uniq_idx, slots = dedup_slots([(tk, sp, did) for _, did, tk, sp, _ in chunk])
        with ThreadPoolExecutor(max_workers=self._vision_workers) as pool:
            decoded = list(pool.map(lambda i: fns[i](docs[i]), uniq_idx))
            if len(decoded) < len(chunk):
                eval_logger.info(
                    "qwen3_5_vllm: {} requests -> {} distinct docs decoded ({:.1f}x saved)",
                    len(chunk),
                    len(decoded),
                    len(chunk) / max(1, len(decoded)),
                )
            # Shared references, not copies: a long clip duplicated K times would multiply
            # peak host memory by K as well.
            visuals = [decoded[i] for i in slots]
            # Per row on the pool too: building a row's message re-encodes each image as a JPEG.
            messages = list(
                pool.map(
                    lambda context, row_visuals: self._build_messages(
                        [context], [row_visuals], self.interleave_visuals
                    )[0],
                    contexts,
                    visuals,
                )
            )
            rows = self._mm_rows(messages, self._render_texts(messages), pool)
        return rows, ids, [tk for _, _, tk, _, _ in chunk]

    @staticmethod
    def _items_from_requests(args: list) -> list[tuple]:
        """(context, doc_id, task, split, doc_to_visual) per request -- see _prepare_chunk."""
        return [(a[0].replace("<image>", ""), a[3], a[4], a[5], a[2]) for a in args]

    def _iter_chunks(self, items: list[tuple], stream: bool = False):
        """Prepare media ahead of generation and yield chunks in request order.

        Streaming starts with small chunks and prepares two ahead. Chunk boundaries depend
        on request position and chunk_size, rather than media-worker timing.
        """
        chunks = split_chunks(
            items, self._chunk_size, STREAM_FIRST_CHUNK if stream else None
        )
        # Track media preparation time that is not overlapped with generation.
        stall_s = 0.0

        def on_ready(ci: int, waited: float) -> None:
            nonlocal stall_s
            stall_s += waited
            eval_logger.info(
                "qwen3_5_vllm rank {}: chunk {}/{} ready, media stall {:.1f}s "
                "(cumulative {:.1f}s over {} chunks, {} rows/chunk)",
                self._rank,
                ci + 1,
                len(chunks),
                waited,
                stall_s,
                ci + 1,
                len(chunks[ci]),
            )

        yield from prefetch(
            chunks, self._prepare_chunk, ahead=2 if stream else 1, on_ready=on_ready
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return run_per_gen_kwargs(
            [reg.args for reg in requests], self._generate_one_config
        )

    def _generate_one_config(self, args: list[tuple]) -> list[str]:
        """Answers for request ``args`` that share one ``generation_kwargs``: the sampling
        params and stop strings below are built once, from the first request."""
        from vllm import SamplingParams

        gen_kwargs = args[0][1]
        items = self._items_from_requests(args)
        until = self._resolve_until(gen_kwargs)

        cur = self._resolve_gen_kwargs(gen_kwargs)
        sp = SamplingParams(
            temperature=cur["temperature"] or 0.0,
            top_p=cur["top_p"] if cur["do_sample"] else 1.0,
            top_k=cur["top_k"] if cur["do_sample"] else -1,
            repetition_penalty=cur["repetition_penalty"],
            max_tokens=cur["max_new_tokens"],
            min_tokens=cur["min_new_tokens"],
            stop=[s for s in until if s],
            # first-token logprobs give p("\n")/p("\n\n") at the think decision point for
            # free -- the HF wrapper needed a whole extra forward pass for this
            logprobs=20 if self._raw_dump_path else None,
        )

        pbar = tqdm(total=len(items), disable=(self.rank != 0), desc="vLLM Responding")
        res: list[str] = []
        raw_records: list[dict] = []
        gen_s = 0.0
        gen_tok = 0
        n_chunks = 0

        for rows, ids, tasks_of in self._iter_chunks(items):
            n_chunks += 1
            t0 = time.perf_counter()
            outs = self._llm.generate(rows, sp)
            elapsed = time.perf_counter() - t0
            gen_s += elapsed
            gen_tok += sum(len(o.outputs[0].token_ids) for o in outs)

            for did, tk, out in zip(ids, tasks_of, outs):
                comp = out.outputs[0]
                raw = comp.text
                ans = self._postprocess_answer(raw, until)
                if self._raw_dump_path:
                    first = {}
                    if comp.logprobs:
                        first = {
                            str(tid): round(float(lp.logprob), 5)
                            for tid, lp in list(comp.logprobs[0].items())[:20]
                        }
                    raw_records.append(
                        {
                            "doc_id": int(did),
                            "task": tk,
                            "has_think": has_reasoning(raw),
                            "raw": raw[:8000],
                            "answer": ans[:400],
                            "n_prompt": len(out.prompt_token_ids or []),
                            "n_gen": [len(comp.token_ids)],
                            "first_token_logprobs": first,
                            # per-chunk generate() time over that chunk's rows, so
                            # batch_gen_s / batch_rows stays an honest per-sample cost
                            "batch_gen_s": round(elapsed, 3),
                            "batch_rows": len(rows),
                        }
                    )
                res.append(ans)
            pbar.update(len(rows))
        pbar.close()

        eval_logger.info(
            "qwen3_5_vllm rank {}: {} prompts in {} chunks, {} generated tokens, "
            "{:.1f}s in generate() => {:.1f} tok/s, {:.2f} s/sample",
            self.rank,
            len(items),
            n_chunks,
            gen_tok,
            gen_s,
            gen_tok / max(gen_s, 1e-9),
            gen_s / max(len(items), 1),
        )
        self._maybe_dump_raw(raw_records)
        return res

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("multi-round is not implemented for qwen3_5_vllm")

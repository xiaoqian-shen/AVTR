"""AVTR adaptive visual imagination and textual reasoning on vLLM.

Uses a checkpoint exported with scripts/bake.sh --emit-markers. Prompts end at a bare
assistant opener so the model can select imagination before the reasoning scaffold.
The query block and gate decisions are processed through chained engine requests.
"""

from __future__ import annotations

from contextlib import closing
from typing import Any

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.imagine.bake import BakeConfigMixin
from lmms_eval.models.model_utils.imagine.chain import (
    ChainRouteMixin,
    check_chain_args,
)
from lmms_eval.models.model_utils.imagine.decode import DecodeMixin

# Imported, not copied: a second copy of the directive text once drifted from training's wording and
# silently fed the model a sentence it had never seen.
from lmms_eval.models.model_utils.imagine.format import (
    as_bool,
    ASSISTANT_OPENER,
    IMAGINE_BLOCK_RE,
    IMAGINE_DIRECTIVE,
    render_block,
    render_decline,
    render_scaffold,
    strip_terminators,
)
from lmms_eval.models.model_utils.imagine.gate import GateMixin
from lmms_eval.models.model_utils.imagine.group import (
    EncodeTallyMixin,
    GroupResult,
)
from lmms_eval.models.qwen3_5_vllm import Qwen3_5VLLM
from loguru import logger as eval_logger
from tqdm import tqdm


@register_model("qwen3_5_imagine_vllm")
class Qwen3_5_ImagineVLLM(
    BakeConfigMixin,
    GateMixin,
    DecodeMixin,
    ChainRouteMixin,
    EncodeTallyMixin,
    Qwen3_5VLLM,
):
    """Adaptive imagine through vLLM on a ``--emit-markers`` vocab bake."""

    def __init__(
        self,
        pretrained: str,
        num_queries: int | None = None,
        # Defaults match training: the imagine directive on every prompt, the block's trailing
        # newline, and the bare </imagine> close on a decline.
        add_directive: bool = True,
        block_tail_newline: bool = True,
        emit_decline: bool = True,
        single_block: bool = True,
        force_imagine_block: bool = False,
        mute_imagine: bool = False,
        imagine_prob_threshold: object = None,
        imagine_gate: str = "think",
        # Read the textual reasoning preference at the token immediately after <think>.
        probe_think: bool = False,
        dump_answer_conf: bool = False,
        think_gate: object = False,
        think_threshold: object = None,
        # Cap the reasoning chain at N tokens; a row that reaches it without "</think>" gets
        # "\n</think>\n\n" appended and one more round for the answer. Without it, a think-on row
        # that loops until max_new_tokens is scored on its truncated chain. Zero disables it.
        think_budget: object = 0,
        # Flush accumulated records periodically to preserve partial-run diagnostics.
        raw_dump_every: int = 200,
        # How much of each generation the dump keeps. A short cap is fine for scoring but cuts a
        # long reasoning chain off mid-thought, so the conclusion, which is the part a route
        # comparison needs, is the part being dropped.
        raw_dump_chars: int = 8000,
        max_new_tokens: int = 1024,
        # Read from model_args rather than from _resolve_gen_kwargs: the default here is greedy while
        # the base no-think default samples at a nonzero temperature, so inheriting it would
        # silently switch this adapter to sampling.
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        # The think probe: single = both raw logprobs from one forward, with a forced fallback for
        # a missing one; pair = two requests, each forced to one id.
        probe_mode: str = "single",
        # How rows are admitted against the encoder cache: auto = budget on media embeds (see
        # imagine/chain.py), always = one row between gate and answer at a time, never = no limit.
        chain_front: str = "auto",
        # Rows rendered and in flight at once; zero means a small multiple of max_num_seqs.
        chain_inflight: int = 0,
        # Encoder cache capacity in embeds. vLLM sizes it to one step's batched tokens, so a large
        # hrbench image or any video fills the half a row may reserve and rows enter one at a time;
        # the larger default costs little memory for encoder outputs.
        encoder_cache_tokens: int = 131072,
        # Largest chunk of the stream (see Qwen3_5VLLM._iter_chunks). It bounds the decoded media
        # held: the chunk being admitted plus the ones prepared ahead stay below what the base
        # class's larger chunks hold.
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self._probe_mode, self._chain_front = check_chain_args(probe_mode, chain_front)
        self._chain_engine_args(kwargs)
        # Engine steps, so decode occupancy can be computed as tokens/step. Without it the only
        # visible number is end-to-end wall clock, which cannot separate "the GPU is doing more
        # work" from "the GPU is idling in a smaller batch".
        self._n_engine_steps: int = 0
        # Monotonic, so the engine-loop request ids stay unique across chunks.
        self._req_seq: int = 0
        super().__init__(pretrained=pretrained, chunk_size=chunk_size, **kwargs)
        self._bake: dict = self._load_bake_config(pretrained)
        self.num_queries: int = int(num_queries or self._bake["num_queries"])
        self._add_directive: bool = as_bool(add_directive, True)
        self._block_tail_newline: bool = as_bool(block_tail_newline, True)
        self._emit_decline: bool = as_bool(emit_decline, True)
        self._single_block: bool = as_bool(single_block, True)
        self._force_block: bool = as_bool(force_imagine_block, False)
        self._mute: bool = as_bool(mute_imagine, False)
        if self._force_block and self._mute:
            raise ValueError(
                "force_imagine_block and mute_imagine are the two opposite brackets; set at most "
                "one (leave both off for the adaptive gate)."
            )
        self._threshold: float | None = self._parse_threshold(imagine_prob_threshold)
        self._raw_dump_every: int = max(0, int(raw_dump_every))
        self._raw_dump_chars: int = max(0, int(raw_dump_chars))
        # Optional answer-token confidence; requires logprobs on every answer request.
        self._answer_conf: bool = as_bool(dump_answer_conf, False)
        # Optional global logit-gap threshold; None selects the gate's argmax.
        self._think_threshold: float | None = (
            None
            if think_threshold is None
            or str(think_threshold).strip().lower() in ("", "none")
            else float(think_threshold)
        )
        self._gate_key: str = (
            "fire_vs_close" if str(imagine_gate) == "close" else "fire_vs_think"
        )
        self._max_num_seqs: int = int(kwargs.get("max_num_seqs", 32) or 32)
        # Where a follow-up's prefix hit ends (the hybrid caches state only at block boundaries),
        # and how much media the encoder cache holds before it evicts.
        vcfg = self._llm.llm_engine.vllm_config
        self._kv_block: int = int(vcfg.cache_config.block_size)
        self._enc_cache: int = self._grow_encoder_cache(int(encoder_cache_tokens))
        self._chain_inflight: int = int(chain_inflight or 0) or 2 * self._max_num_seqs
        self._join_checked: bool = False
        self._gate_missing: int = 0
        self._install_encode_tally()
        if not hasattr(self._llm, "_preprocess_cmpl"):
            raise RuntimeError(
                "qwen3_5_imagine_vllm renders each row once with LLM._preprocess_cmpl, which this "
                "vLLM build does not have"
            )
        eval_logger.info(
            "imagine_vllm rank {}: KV block {}, encoder cache {}",
            self._rank,
            self._kv_block,
            self._enc_cache,
        )
        self.imagine_max_new_tokens: int = int(max_new_tokens)
        self._temperature: float = float(temperature)
        self._top_p: float = float(top_p)
        self._top_k: int = int(top_k)
        # One block per turn is what training renders; without the cap a model that reaches its
        # localization step mid-reasoning emits block after block and never reaches </think>.
        self._max_blocks: int = 1 if self._single_block else 4

        self._open_id, self._close_id = self._marker_ids()
        self._think_id: int = int(self.tokenizer.convert_tokens_to_ids("<think>"))
        self._think_end_id: int = int(self.tokenizer.convert_tokens_to_ids("</think>"))
        self._probe_think: bool = bool(probe_think)
        self._think_budget: int = int(think_budget)
        self._think_gate: bool = str(think_gate).lower() not in ("false", "0", "")
        if self._think_gate and not self._probe_think:
            # The probe IS the gate: without it t_confs is empty, every argmax_reason reads False,
            # and the run silently becomes no-think everywhere while reporting an adaptive route.
            raise ValueError("think_gate=True requires probe_think=True")
        # Resolved from the tokenizer rather than hard-coded: a retokenized checkpoint would
        # silently probe two unrelated ids and report a confident, meaningless gate.
        self._nl_id: int = int(
            self.tokenizer("\n", add_special_tokens=False)["input_ids"][0]
        )
        self._nl2_id: int = int(
            self.tokenizer("\n\n", add_special_tokens=False)["input_ids"][0]
        )
        if self._probe_think and self._nl_id == self._nl2_id:
            # "\n\n" tokenized to two "\n" rather than its own id. The gap would be identically zero
            # and read as a perfectly balanced gate on every question.
            if self._think_gate:
                # Every row would silently take the no-think route under a run that reports an
                # adaptive one.
                raise ValueError(
                    f"think_gate needs distinct ids for '\\n' and '\\n\\n'; both are {self._nl_id}"
                )
            eval_logger.warning(
                "think probe DISABLED: '\\n' and '\\n\\n' share id {}", self._nl_id
            )
            self._probe_think = False
        self._query_ids: list[int] = self._resolve_query_ids()
        self._tool_ids: list[int] = self._resolve_tool_ids()
        self._require_emittable_markers()

        self._block_text: str = render_block(
            self.num_queries, self._block_tail_newline, self._bake["query_token_format"]
        )
        self._decline_text: str = (
            render_decline(self._block_tail_newline) if self._emit_decline else ""
        )
        self._scaffold: str = render_scaffold(self.enable_thinking)
        eval_logger.info(
            "imagine_vllm: route={} probe_mode={} chain_front={} num_queries={} directive={} "
            "tail_nl={} emit_decline={} single_block={} threshold={} gate={} "
            "enable_thinking={} temperature={}",
            self._route_name(),
            self._probe_mode,
            self._chain_front,
            self.num_queries,
            self._add_directive,
            self._block_tail_newline,
            self._emit_decline,
            self._single_block,
            self._threshold,
            self._gate_key,
            self.enable_thinking,
            self._temperature,
        )

    @staticmethod
    def _chain_engine_args(kwargs: dict) -> None:
        """Engine settings the chain depends on: set, not defaulted, because an explicit override
        breaks it silently. A nonzero processor cache returns data=None for items it believes the
        engine holds, which a resubmitted rendered input then carries into an assert; fcfs makes
        every request priority inert, so follow-ups queue behind new rows and lose their prefix."""
        cache = kwargs.get("mm_processor_cache_gb", 0)
        if cache is None or float(cache) != 0:
            raise ValueError("qwen3_5_imagine_vllm needs mm_processor_cache_gb=0")
        if not as_bool(kwargs.get("scheduling_priority", True), True):
            raise ValueError("qwen3_5_imagine_vllm needs scheduling_priority=True")
        kwargs["mm_processor_cache_gb"] = 0
        kwargs["scheduling_priority"] = True

    def _grow_encoder_cache(self, tokens: int) -> int:
        """Raise the scheduler's encoder cache to ``tokens`` embeds and return its capacity. The
        model runner stores encoder outputs as they come, so only the scheduler's accounting has a
        size."""
        try:
            core = self._llm.llm_engine.engine_core.engine_core
            mgr = core.scheduler.encoder_cache_manager
        except Exception:  # noqa: BLE001 -- a private path must never fail the eval
            cfg = self._llm.llm_engine.vllm_config.scheduler_config
            return int(getattr(cfg, "encoder_cache_size", 0) or 0)
        extra = tokens - int(mgr.cache_size)
        if extra > 0:
            mgr.cache_size += extra
            mgr.num_free_slots += extra
            mgr.num_freeable_slots += extra
        return int(mgr.cache_size)

    # ---- prompt ---------------------------------------------------------------------------

    def _build_messages(self, contexts, visual_list, interleave_visuals):
        if self._add_directive:
            contexts = [f"{IMAGINE_DIRECTIVE}{c}" for c in contexts]
        return super()._build_messages(contexts, visual_list, interleave_visuals)

    def _render_texts(self, batched_messages) -> list[str]:
        """End prompts at a bare assistant opener, before the imagination decision."""
        return [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            + ASSISTANT_OPENER
            for msg in batched_messages
        ]

    # ---- generation -------------------------------------------------------------------------

    def _generate_one_config(self, args: list[tuple]) -> list[str]:
        """Generate one request group with a shared stopping rule and token budget."""
        gen_kwargs = args[0][1]
        items = self._items_from_requests(args)
        until = self._resolve_until(gen_kwargs)
        budget = int(gen_kwargs.get("max_new_tokens", self.imagine_max_new_tokens))

        pbar = tqdm(total=len(items), disable=(self.rank != 0), desc="Imagine vLLM")
        res: list[str] = []
        raw_records: list[dict] = []
        dumped = 0  # records already flushed to raw_dump_path
        stats = {
            "s": 0.0,
            "tok": 0,
            "fired": 0,
            "n": 0,
            "late": 0,
            "forced": 0,
            "steps": 0,
            "encodes": 0,
        }

        # closing(): a failure in this loop must still abort the stream's requests in the engine.
        with closing(
            self._chain_stream(self._iter_chunks(items, stream=True), budget, until)
        ) as stream:
            for gr, ids, tasks_of in stream:
                self._account(stats, gr)
                for k, (did, tk) in enumerate(zip(ids, tasks_of)):
                    ans = self._postprocess_answer(gr.gen[k], until)
                    res.append(ans)
                    if self._raw_dump_path:
                        raw_records.append(self._record(gr, k, did, tk, ans, len(ids)))
                if (
                    self._raw_dump_every
                    and self._raw_dump_path
                    and len(raw_records) - dumped >= self._raw_dump_every
                ):
                    self._maybe_dump_raw(raw_records)
                    dumped = len(raw_records)
                pbar.update(len(ids))
                if stats["n"] % 512 < len(ids):
                    # Periodic, for the same reason as the stall log: the end-of-run summary never
                    # prints for a run that is killed or preempted.
                    eval_logger.info(
                        "imagine_vllm rank {}: {} rows | {:.1f}s ({:.2f}s/row gen) | fired {:.0%} | "
                        "{:.0f} tok/row",
                        self.rank,
                        stats["n"],
                        stats["s"],
                        stats["s"] / max(1, stats["n"]),
                        stats["fired"] / max(1, stats["n"]),
                        stats["tok"] / max(1, stats["n"]),
                    )
        pbar.close()
        self._finish_run(stats)
        self._maybe_dump_raw(raw_records)
        return res

    @staticmethod
    def _account(stats: dict, gr: GroupResult) -> None:
        stats["s"] += gr.wall_s
        stats["n"] += len(gr.gen)
        stats["fired"] += sum(gr.fires)
        stats["late"] += sum(b - int(f) for b, f in zip(gr.blocks, gr.fires))
        stats["tok"] += sum(gr.ngen)
        stats["forced"] += len(gr.forced)
        stats["steps"] += gr.steps
        stats["encodes"] += gr.encodes

    def _finish_run(self, stats: dict) -> None:
        """End-of-run summary logs, before the final raw-dump flush."""
        self._log_summary(stats)
        if self._think_budget > 0:
            eval_logger.info(
                "imagine_vllm rank {}: think_budget={} force-closed {}/{} chains ({:.1%})",
                self.rank,
                self._think_budget,
                stats["forced"],
                stats["n"],
                stats["forced"] / max(stats["n"], 1),
            )
        if self._gate_missing:
            eval_logger.warning(
                "imagine_vllm rank {}: {} gate reads had no {} value and declined",
                self.rank,
                self._gate_missing,
                self._gate_key,
            )

    def _record(self, gr: GroupResult, k: int, did, task, ans: str, rows: int) -> dict:
        """One raw-dump row. ``batch_rows`` is the number of rows in this row's chunk.

        ``batch_gen_s`` is the time from the previous chunk's completion to this row's chunk's, so
        the chunks of a rank add up to its generation time as in the ``qwen3_5_vllm`` dumps; the row's
        own timeline is under ``chain``.
        """
        raw, fired, n_blocks = gr.gen[k], gr.fires[k], gr.blocks[k]
        # Everything after "<think>", whether or not the row closed it. Requiring "</think>" made
        # an unclosed chain indistinguishable from an empty one, so a row that reasoned was
        # reported as a no-reasoning row -- and the two arms of the gate were measured against the
        # wrong partition. Closure is a format property and is reported separately.
        head = raw.split("</think>", 1)[0]
        reasoning = head.rsplit("<think>", 1)[-1] if "<think>" in head else ""
        closed_think = "</think>" in raw
        return {
            "doc_id": int(did),
            "task": task,
            "fired": bool(fired),
            "n_blocks": int(n_blocks),
            "late_block": int(n_blocks) > int(bool(fired)),
            # Mirrors the training reward: an unclosed row's text runs chain and answer together,
            # so it needs a floor, while a closed one is delimited and any content counts. The
            # floor sits in the empty gap between the measured lengths of real chains and of bare
            # answer fields.
            "has_think": bool(len(reasoning.strip()) > (0 if closed_think else 200)),
            "closed_think": closed_think,
            "forced_close": k in gr.forced,
            "think_decided": None if gr.thinks is None else bool(gr.thinks[k]),
            "n_tokens": int(gr.ngen[k]),
            "n_gen": [int(gr.ngen[k])],
            "truncated": bool(gr.trunc[k]),
            "conf": gr.confs[k],
            "think_conf": gr.t_confs[k] or {},
            "answer_conf": gr.answer_conf.get(k) or {},
            "raw": raw[: self._raw_dump_chars] if self._raw_dump_chars else raw,
            "answer": ans[:400],
            "batch_gen_s": round(gr.wall_s, 3),
            "batch_rows": rows,
            "group_encodes": gr.encodes,
            **gr.extra[k],
        }

    def _log_summary(self, stats: dict) -> None:
        n = max(stats["n"], 1)
        total = stats["s"]
        steps = stats["steps"]
        eval_logger.info(
            "imagine_vllm rank {}: route={} fired {}/{} ({:.1%}), {} late blocks; "
            "{} tokens in {:.1f}s => {:.1f} tok/s, {:.2f} s/sample; "
            "{} engine steps, {} tok/step, {} media encodes",
            self.rank,
            self._route_name(),
            stats["fired"],
            stats["n"],
            stats["fired"] / n,
            stats["late"],
            stats["tok"],
            total,
            stats["tok"] / max(total, 1e-9),
            total / n,
            steps,
            # Decode occupancy. max_num_seqs is the ceiling; well under it means the batch is
            # running part-empty, which end-to-end wall clock cannot distinguish from real work.
            f"{stats['tok'] / steps:.2f}" if steps else "n/a",
            stats["encodes"],
        )

    def _postprocess_answer(self, ans, until):
        """Remove imagination tokens and the reasoning scaffold before answer extraction."""
        ans = IMAGINE_BLOCK_RE.sub("", strip_terminators(ans))
        _, think_end, remaining = ans.partition("</think>")
        if think_end:
            ans = remaining.strip()
        return super()._postprocess_answer(ans, until)

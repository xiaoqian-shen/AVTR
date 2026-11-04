"""Chain visual gates, reasoning gates, and answers on one vLLM engine loop.

Follow-up requests have priority over new rows. Media reservations bound admission
against the encoder cache, and rows are rendered once as chunks enter the stream.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from lmms_eval.models.model_utils.imagine.format import (
    IMAGINE_OPEN,
    IMAGINE_THINK_PREFIX,
    THINK_FORCE_ANSWER_MIN,
    THINK_FORCE_CLOSE,
)
from lmms_eval.models.model_utils.imagine.group import GroupResult
from loguru import logger as eval_logger

# (rows, doc ids, tasks) of one chunk, as Qwen3_5VLLM._iter_chunks yields them.
Chunk = tuple[list[dict], list, list]

# vLLM orders the waiting queue by (priority, arrival_time), smaller first.
PRIO_FOLLOW = 0
PRIO_FIRST = 1
# Consecutive engine steps that return nothing for any live request before the loop gives up. A
# long prefill legitimately returns nothing for a few steps; far more means a wedged engine.
CHAIN_IDLE_STEPS = 10000


def check_chain_args(probe_mode: object, chain_front: object) -> tuple[str, str]:
    """``(probe_mode, chain_front)`` as strings, or a ValueError for a value the chain cannot run."""
    probe_mode, chain_front = str(probe_mode), str(chain_front)
    if probe_mode not in ("pair", "single"):
        raise ValueError(f"probe_mode must be pair or single, not {probe_mode!r}")
    if chain_front not in ("auto", "always", "never"):
        raise ValueError(
            f"chain_front must be auto, always or never, not {chain_front!r}"
        )
    return probe_mode, chain_front


def media_span(placeholders, hit: int = 0) -> tuple[int, int, int]:
    """(end of the last media placeholder, total embeds, embeds of the items a follow-up whose
    prefix hit ends at ``hit`` must encode again) from rendered ``mm_placeholders``. vLLM encodes
    whole items, so an item the tail only partly reaches counts in full."""
    end = embeds = tail = 0
    for ranges in (placeholders or {}).values():
        for ph in ranges:
            stop = int(ph.offset) + int(ph.length)
            n = int(ph.get_num_embeds())
            end = max(end, stop)
            embeds += n
            if stop > hit:
                tail += n
    return end, embeds, tail


@dataclass
class ChainRow:
    """One row's state. The decode fields mean what ``GroupResult``'s per-row lists mean."""

    i: int  # index within its chunk
    key: int  # unique across the stream
    row: dict | None
    inputs: dict | None = None
    base_ids: list[int] = field(default_factory=list)
    media_end: int = 0
    media_embeds: int = 0
    tail_embeds: int = 0
    needs_media: bool = False
    reserve: int = 0
    media_done: bool = False
    fired: bool = False
    conf: dict = field(default_factory=dict)
    lps: dict[int, float] = field(default_factory=dict)
    probe_left: int = 0
    probe_rounds: int = 0
    probe_fallback: int = 0
    probe_done: bool = False
    think_conf: dict = field(default_factory=dict)
    think: bool | None = None
    gen: str = ""
    ngen: int = 0
    used: int = 0
    blocks: int = 0
    masked: bool = False
    trunc: bool = False
    rounds: int = 0
    forced: bool = False
    answer_conf: dict | None = None
    # The DELTA outputs of the answer or force-close request in flight.
    acc_text: str = ""
    acc_n: int = 0
    acc_first: bool = True
    outstanding: int = 0
    done: bool = False
    render_s: float = 0.0
    rids: list[str] = field(default_factory=list)
    # Per stage, one entry per request: [submitted, (first output,) finished], seconds since the
    # group started, and num_cached_tokens.
    t: dict = field(default_factory=dict)
    cached: dict = field(default_factory=dict)


@dataclass
class ChainGroup:
    tag: int
    t0: float
    budget: int
    cap: int
    until: list[str]
    # request id -> (kind, row, meta); meta is the probe's owned ids.
    live: dict = field(default_factory=dict)
    # Rows (by key) between their first request and the last request that needs their media, and
    # the encoder embeds they hold between them.
    fronts: set[int] = field(default_factory=set)
    reserved: int = 0
    rows_in_flight: int = 0
    rows_seen: int = 0
    steps: int = 0
    idle: int = 0  # consecutive steps without output
    seq: int = 0
    # Clock and step count when the previous chunk was returned: chunks overlap, so each is charged
    # the time from the previous one's completion to its own.
    mark_t: float = 0.0
    mark_steps: int = 0
    mark_preemptions: int = 0


class ChainRouteMixin:
    """Group methods of ``Qwen3_5_ImagineVLLM``, whose ``__init__`` sets their state."""

    def _chain_stream(
        self, chunks: Iterable[Chunk], budget: int, until: list[str]
    ) -> Iterator[tuple[GroupResult, list, list]]:
        """``(GroupResult, ids, tasks)`` per chunk, in order, from one engine loop."""
        eng = self._llm.llm_engine
        self._req_seq += 1
        g = ChainGroup(
            tag=self._req_seq,
            t0=time.perf_counter(),
            budget=budget,
            # With think_budget the answer rounds decode against the cap, and the force-close gets
            # what is left of max_new_tokens.
            cap=min(budget, self._think_budget) if self._think_budget > 0 else budget,
            until=until,
        )
        g.mark_t = g.t0
        handlers = {
            "g": self._chain_on_gate,
            "p": self._chain_on_probe,
            "a": self._chain_on_answer,
            "f": self._chain_on_force_close,
        }
        source = iter(chunks)
        pending: deque[ChainRow] = deque()
        open_chunks: deque[tuple[list[ChainRow], list, list]] = deque()
        exhausted = False
        try:
            while True:
                if not pending and not exhausted:
                    exhausted = not self._chain_take(source, g, pending, open_chunks)
                self._chain_admit(eng, pending, g)
                while open_chunks and all(st.done for st in open_chunks[0][0]):
                    states, ids, tasks = open_chunks.popleft()
                    yield self._chain_result(states, g), ids, tasks
                if exhausted and not open_chunks:
                    return
                if g.live:
                    self._chain_step(eng, g, handlers)
                elif pending:
                    raise RuntimeError(
                        "chain: rows are pending but nothing is in flight (admission stalled)"
                    )
                elif exhausted:
                    raise RuntimeError("chain: the engine drained with rows unfinished")
        except BaseException:
            # Requests left in the engine would reach the next LLM.generate(), which sorts its
            # outputs by int(request_id) and cannot parse these ids.
            if g.live:
                eng.abort_request(list(g.live))
            raise

    def _chain_take(
        self, source, g: ChainGroup, pending: deque, open_chunks: deque
    ) -> bool:
        """Queue the next chunk's rows; False once ``source`` is exhausted."""
        chunk = next(source, None)
        if chunk is None:
            return False
        rows, ids, tasks = chunk
        states = []
        for i, row in enumerate(rows):
            states.append(ChainRow(i, g.rows_seen, row))
            g.rows_seen += 1
        open_chunks.append((states, ids, tasks))
        pending.extend(states)
        return True

    def _chain_step(self, eng, g: ChainGroup, handlers: dict) -> None:
        """One engine step, routing each output to the stage handler of its request."""
        g.steps += 1
        self._n_engine_steps += 1
        outs = eng.step()
        g.idle = 0 if outs else g.idle + 1
        if g.idle > CHAIN_IDLE_STEPS:
            raise RuntimeError(
                f"chain: {g.idle} engine steps without output, {len(g.live)} requests live"
            )
        for out in outs:
            entry = g.live.get(out.request_id)
            if entry is None:
                continue
            kind, st, meta = entry
            if out.finished:
                del g.live[out.request_id]
                st.outstanding -= 1
            handlers[kind](eng, out, st, g, meta)
            self._chain_maybe_finish(st, g)

    # ---- admission and rendering -------------------------------------------------------------

    def _chain_render(self, st: ChainRow) -> None:
        t0 = time.perf_counter()
        (inputs,) = self._llm._preprocess_cmpl([st.row])
        st.render_s = time.perf_counter() - t0
        st.inputs = dict(inputs)
        st.base_ids = list(inputs["prompt_token_ids"])
        # A follow-up's prefix hit ends at the last block boundary inside the prompt (the hybrid
        # caches state only there); its tail needs the encoder output of every item it reaches.
        hit = (len(st.base_ids) // self._kv_block) * self._kv_block
        st.media_end, st.media_embeds, st.tail_embeds = media_span(
            inputs.get("mm_placeholders"), hit
        )
        st.needs_media = st.tail_embeds > 0
        if not self._join_checked:
            self._chain_check_join(st)
            self._join_checked = True

    def _chain_check_join(self, st: ChainRow) -> None:
        """Follow-ups append continuation ids to the rendered prompt ids instead of rendering the
        joined text, which is only the same thing if the tokenizer does not merge across the
        junction, and if the tokenizer that counts continuations is vLLM's. Checked once."""
        for fired in (True, False):
            text = self._probe_text(fired)
            (joined,) = self._llm._preprocess_cmpl(
                [{**st.row, "prompt": st.row["prompt"] + text}]
            )
            if list(joined["prompt_token_ids"]) != st.base_ids + self._ids(text):
                raise RuntimeError(
                    "chain: prompt + continuation does not tokenize as prompt ids + continuation "
                    f"ids (fired={fired}); the chain cannot reuse the rendered prompt"
                )
        sample = (
            self._block_text + IMAGINE_THINK_PREFIX + "\nreasoning" + THINK_FORCE_CLOSE
        )
        theirs = self._llm.get_tokenizer()(sample, add_special_tokens=False)[
            "input_ids"
        ]
        if self._ids(sample) != list(theirs):
            raise RuntimeError("chain: the adapter's tokenizer and vLLM's disagree")

    def _chain_fits(self, st: ChainRow, g: ChainGroup) -> bool:
        if not g.fronts or self._chain_front == "never":
            return True
        if self._chain_front == "always":
            return False
        return g.reserved + st.media_embeds <= self._enc_cache // 2

    def _chain_admit(self, eng, pending: deque, g: ChainGroup) -> None:
        while pending and g.rows_in_flight < self._chain_inflight:
            st = pending[0]
            if st.inputs is None:
                self._chain_render(st)
            # In order: a row that does not fit waits rather than being overtaken.
            if not self._chain_fits(st, g):
                return
            pending.popleft()
            g.rows_in_flight += 1
            g.fronts.add(st.key)
            self._chain_reserve(st, g, st.media_embeds)
            if self._force_block or self._mute:
                st.fired = self._force_block
                self._chain_after_gate(eng, st, g, PRIO_FIRST)
                self._chain_maybe_finish(st, g)
            else:
                self._chain_add(
                    eng,
                    "g",
                    st,
                    [],
                    self._final_only(self._gate_params()),
                    PRIO_FIRST,
                    g,
                )

    def _chain_add(self, eng, kind, st, cont_ids, params, prio, g, meta=None) -> None:
        g.seq += 1
        rid = f"{kind}{g.tag}_{st.key}_{g.seq}"
        inputs = st.inputs
        if cont_ids:
            inputs = {
                k: v
                for k, v in st.inputs.items()
                if k not in ("prompt", "arrival_time")
            }
            inputs["prompt_token_ids"] = st.base_ids + list(cont_ids)
        try:
            eng.add_request(
                rid, inputs, params, arrival_time=time.time(), priority=prio
            )
        except BaseException:
            # add_request registers the request for output before the engine core rejects it,
            # which would leave it unfinished forever.
            eng.abort_request([rid])
            raise
        g.live[rid] = (kind, st, meta)
        st.rids.append(rid)
        st.outstanding += 1
        st.t.setdefault(kind, []).append([round(time.perf_counter() - g.t0, 3)])

    def _chain_reserve(self, st: ChainRow, g: ChainGroup, embeds: int) -> None:
        g.reserved += embeds - st.reserve
        st.reserve = embeds

    def _chain_leave_front(self, st: ChainRow, g: ChainGroup) -> None:
        g.fronts.discard(st.key)
        self._chain_reserve(st, g, 0)

    def _chain_media_done(self, st: ChainRow, g: ChainGroup) -> None:
        """The row's first request has passed its media: from here only its tail needs it."""
        if st.media_done:
            return
        st.media_done = True
        if st.needs_media:
            self._chain_reserve(st, g, min(st.reserve, st.tail_embeds))
        else:
            self._chain_leave_front(st, g)

    def _chain_maybe_finish(self, st: ChainRow, g: ChainGroup) -> None:
        """A row is done once nothing of it is in flight: every handler sends the next stage
        before returning, so an idle row has no stage left."""
        if st.outstanding or st.done:
            return
        st.done = True
        st.inputs = st.row = None
        g.rows_in_flight -= 1
        self._chain_leave_front(st, g)

    def _chain_mark(self, st: ChainRow, kind: str, out, g: ChainGroup) -> None:
        st.t[kind][-1].append(round(time.perf_counter() - g.t0, 3))
        if len(st.t[kind][-1]) == 2:
            st.cached.setdefault(kind, []).append(
                getattr(out, "num_cached_tokens", None)
            )

    # ---- stages ------------------------------------------------------------------------------

    def _chain_on_gate(self, eng, out, st: ChainRow, g: ChainGroup, meta) -> None:
        self._chain_mark(st, "g", out, g)
        st.conf = self._gate_conf(out.outputs[0])
        st.fired = self._decide(st.conf)
        self._chain_media_done(st, g)
        self._chain_after_gate(eng, st, g, PRIO_FOLLOW)

    def _chain_after_gate(self, eng, st: ChainRow, g: ChainGroup, prio: int) -> None:
        if self._probe_think:
            first = (
                [((self._nl_id, self._nl2_id), self._think_probe_single_params())]
                if self._probe_mode == "single"
                else [((self._nl_id,), self._think_probe_params(self._nl_id))]
            )
            self._chain_probe(eng, st, g, prio, first)
            if self._think_gate:
                return
        else:
            st.probe_done = True
        # Without think_gate the probe only measures: the answer keeps the fixed scaffold and does
        # not wait for it.
        self._chain_answer(eng, st, g, prio)

    def _chain_probe(self, eng, st: ChainRow, g: ChainGroup, prio: int, reqs) -> None:
        """think_conf = sigmoid(logit(\\n) - logit(\\n\\n)) right after ``<think>``: on a positive
        gap the model would open reasoning, on a negative one it would emit the empty-think format.
        The prefix is the one the answer continues from -- block or decline, then ``<think>`` -- or
        the probe measures a state the run never enters."""
        ids = self._ids(self._probe_text(st.fired))
        st.probe_left += len(reqs)
        for owned, params in reqs:
            self._chain_add(eng, "p", st, ids, self._final_only(params), prio, g, owned)

    def _chain_on_probe(self, eng, out, st: ChainRow, g: ChainGroup, owned) -> None:
        """A pair request is read only for the id it forced; the single request owns both ids."""
        self._chain_mark(st, "p", out, g)
        self._chain_media_done(st, g)  # a forced route has no gate: the probe was first
        lp = self._raw_lps(out.outputs[0])
        for tid in owned:
            if tid in lp:
                st.lps[tid] = lp[tid]
        st.probe_left -= 1
        if st.probe_left:
            return
        st.probe_rounds += 1
        if st.probe_rounds == 1 and self._probe_mode == "pair":
            reqs = [((self._nl2_id,), self._think_probe_params(self._nl2_id))]
            self._chain_probe(eng, st, g, PRIO_FOLLOW, reqs)
            return
        missing = [t for t in (self._nl_id, self._nl2_id) if t not in st.lps]
        if st.probe_rounds == 1 and missing:
            # Never seen on the checkpoint this harness was validated on, where the pair is the two most
            # likely tokens of every row; kept so a checkpoint where it is not still gets both values
            # rather than an empty think_conf.
            st.probe_fallback = len(missing)
            reqs = [((t,), self._think_probe_params(t)) for t in missing]
            self._chain_probe(eng, st, g, PRIO_FOLLOW, reqs)
            return
        st.probe_done = True
        st.think_conf = self._think_conf_from_lps(
            st.lps.get(self._nl_id), st.lps.get(self._nl2_id)
        )
        if self._think_gate:
            st.think = self._decide_think(st.think_conf)
            self._chain_answer(eng, st, g, PRIO_FOLLOW)
        else:
            self._chain_leave_front(st, g)

    def _chain_answer(self, eng, st: ChainRow, g: ChainGroup, prio: int) -> None:
        cont = self._continuation(st.fired, st.think if self._think_gate else None)
        st.gen, st.used = cont, self._n_tok(cont)
        st.blocks, st.masked = int(st.fired), st.fired and self._single_block
        # Sent at PRIO_FOLLOW, the answer is scheduled before any row that starts after this. A
        # measurement-only probe that is still owed keeps the reservation until it returns.
        if st.probe_done:
            self._chain_leave_front(st, g)
        if g.cap - st.used <= 0:
            return
        st.rounds = 1
        params = self._answer_params(g.cap - st.used, g.until, st.masked)
        self._chain_add(eng, "a", st, self._ids(cont), params, prio, g)

    def _chain_take_delta(self, out, st: ChainRow, kind: str, g: ChainGroup) -> bool:
        """Accumulate one DELTA output; True once the request has finished."""
        comp = out.outputs[0]
        if st.acc_first:
            # The first output of a request comes when its prefill is done.
            self._chain_mark(st, kind, out, g)
            if kind == "a" and st.rounds == 1 and self._answer_conf:
                st.answer_conf = self._first_token_conf(comp)
            st.acc_first = False
        st.acc_text += comp.text
        st.acc_n += len(comp.token_ids)
        if not out.finished:
            return False
        self._chain_mark(st, kind, out, g)
        st.gen += st.acc_text
        st.ngen += st.acc_n
        st.used += st.acc_n
        st.trunc = comp.finish_reason == "length"
        st.acc_text, st.acc_n, st.acc_first = "", 0, True
        return True

    def _chain_on_answer(self, eng, out, st: ChainRow, g: ChainGroup, meta) -> None:
        if not self._chain_take_delta(out, st, "a", g):
            return
        reopened = self._chain_reinject(st)
        if reopened and g.cap > st.used and st.rounds < self._max_blocks + 1:
            st.rounds += 1
            params = self._answer_params(g.cap - st.used, g.until, st.masked)
            self._chain_add(eng, "a", st, self._ids(st.gen), params, PRIO_FOLLOW, g)
            return
        if (
            self._think_budget > 0
            and st.trunc
            and "</think>" not in st.gen.rsplit("<think>", 1)[-1]
        ):
            st.gen += THINK_FORCE_CLOSE
            st.used += self._n_tok(THINK_FORCE_CLOSE)
            st.forced = True
            # The answer gets what the chain left of max_new_tokens, but never less than the floor:
            # with think_budget == max_new_tokens nothing would be left to answer with.
            params = self._answer_params(
                max(g.budget - st.used, THINK_FORCE_ANSWER_MIN), g.until, st.masked
            )
            self._chain_add(eng, "f", st, self._ids(st.gen), params, PRIO_FOLLOW, g)

    def _chain_on_force_close(
        self, eng, out, st: ChainRow, g: ChainGroup, meta
    ) -> None:
        self._chain_take_delta(out, st, "f", g)

    def _chain_reinject(self, st: ChainRow) -> bool:
        """Complete a block the model just opened. True if it did."""
        if not st.gen.endswith(IMAGINE_OPEN):
            return False
        body = self._block_text[len(IMAGINE_OPEN) :]
        st.gen += body
        st.used += self._n_tok(body)
        st.blocks += 1
        if self._single_block:
            st.masked = True
        return True

    # ---- helpers -----------------------------------------------------------------------------

    def _probe_text(self, fired: bool) -> str:
        return (
            self._block_text if fired else self._decline_text
        ) + IMAGINE_THINK_PREFIX

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _final_only(params):
        from vllm.sampling_params import RequestOutputKind

        params.output_kind = RequestOutputKind.FINAL_ONLY
        return params

    def _chain_result(self, states: list[ChainRow], g: ChainGroup) -> GroupResult:
        if not (self._force_block or self._mute) and any(not st.conf for st in states):
            raise RuntimeError("chain: a row was answered without its gate being read")
        now = time.perf_counter()
        preempted = getattr(self, "_preemptions", 0)
        wall, steps = now - g.mark_t, g.steps - g.mark_steps
        preempted, g.mark_preemptions = preempted - g.mark_preemptions, preempted
        g.mark_t, g.mark_steps = now, g.steps
        tokens = sum(st.ngen for st in states)
        encodes = {st.i: sum(self._take_encodes(r) for r in st.rids) for st in states}
        empty = sum(not st.think_conf for st in states) if self._probe_think else 0
        if empty:
            # Silence here is how a whole checkpoint once came back with think_conf={} on every
            # row, every row routed to no-think, and no way to tell an unset flag from a vLLM that
            # returned the wrong logprob keys.
            eval_logger.warning(
                "think probe: {}/{} rows produced no gate value (want ids nl={} nl2={})",
                empty,
                len(states),
                self._nl_id,
                self._nl2_id,
            )
        eval_logger.info(
            "imagine_vllm rank {}: chain group of {} rows in {:.1f}s, {} engine steps "
            "({:.2f} tok/step), render {:.1f}s, {} media encodes, {} preemptions, {} probe fallbacks",
            self.rank,
            len(states),
            wall,
            steps,
            tokens / max(steps, 1),
            sum(st.render_s for st in states),
            sum(encodes.values()),
            preempted,
            sum(bool(st.probe_fallback) for st in states),
        )
        return GroupResult(
            gen=[st.gen for st in states],
            ngen=[st.ngen for st in states],
            blocks=[st.blocks for st in states],
            trunc=[st.trunc for st in states],
            fires=[st.fired for st in states],
            confs=[st.conf for st in states],
            t_confs=[st.think_conf for st in states],
            thinks=[bool(st.think) for st in states] if self._think_gate else None,
            forced={st.i for st in states if st.forced},
            answer_conf={
                st.i: st.answer_conf for st in states if st.answer_conf is not None
            },
            wall_s=wall,
            steps=steps,
            encodes=sum(encodes.values()),
            extra=[
                {
                    "chain": {
                        "render_s": round(st.render_s, 3),
                        "t": st.t,
                        "cached": st.cached,
                        "encodes": encodes[st.i],
                        "n_prompt": len(st.base_ids),
                        "media_end": st.media_end,
                        "media_embeds": st.media_embeds,
                        "tail_embeds": st.tail_embeds,
                        "probe_fallback": st.probe_fallback,
                    }
                }
                for st in states
            ],
        )

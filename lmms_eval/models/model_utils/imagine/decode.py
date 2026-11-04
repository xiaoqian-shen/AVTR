"""The answer after the gate: the run injected after the gate, and the sampling params of the free
decode that follows it."""

from __future__ import annotations

from lmms_eval.models.model_utils.imagine.format import IMAGINE_OPEN


class DecodeMixin:
    """Decode methods of ``Qwen3_5_ImagineVLLM``, whose ``__init__`` sets their state."""

    # ---- sampling params ------------------------------------------------------------------

    def _answer_params(self, max_tokens: int, until: list[str], markers_masked: bool):
        """Free decode after the injected run.

        ``single_block`` masks BOTH markers once a block exists: banning only OPEN moves the
        degeneration rather than removing it (the model emits a run of bare CLOSE instead). While
        they are live, OPEN is a stop string so a mid-answer trigger gets its block injected rather
        than free-decoded.

        DELTA outputs, which the chain accumulates (``_chain_take_delta``) so it can time the
        prefill and read the first token; ``LLM.generate()`` would override it with FINAL_ONLY.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        ids = list(self._query_ids) + list(self._tool_ids)
        if markers_masked:
            ids += [self._open_id, self._close_id]
        elif self._mute:
            ids.append(self._open_id)
        stop = [s for s in until if s]
        if self._open_id not in ids:
            stop = stop + [IMAGINE_OPEN]
        sample = self._temperature > 0
        return SamplingParams(
            temperature=self._temperature if sample else 0.0,
            top_p=self._top_p if sample else 1.0,
            top_k=self._top_k if sample and self._top_k > 0 else -1,
            max_tokens=max(1, max_tokens),
            stop=stop,
            logprobs=5 if self._answer_conf else None,
            logit_bias=self._mask(ids),
            # <imagine> and the query tokens are ADDED SPECIAL tokens, so the vLLM default
            # (skip_special_tokens=True) would erase the block from the text -- taking the OPEN stop
            # string with it and making every fired sample look like a decline.
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            output_kind=RequestOutputKind.DELTA,
        )

    # ---- continuation ---------------------------------------------------------------------

    def _n_tok(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _continuation(self, fired: bool, think: bool | None = None) -> str:
        """The deterministic run injected after the gate, before free decode.

        ``think=None`` keeps the fixed scaffold chosen by ``enable_thinking``. Passing it makes the
        reason/do-not-reason decision PER ROW, which is the only way the model's own choice reaches
        the output: render_scaffold prefills the token after ``<think>``, so without it the
        reasoning decision is the harness's, not the model's.
        """
        scaffold = (
            self._scaffold
            if think is None
            else ("<think>\n" if think else "<think>\n\n</think>\n\n")
        )
        return (self._block_text if fired else self._decline_text) + scaffold

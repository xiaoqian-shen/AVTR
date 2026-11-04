"""Generate a rollout from an injected route prefix and score its answer.

Handles empty-chain redraws, forced reasoning closure, and token loss masks.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Collection, Sequence
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

from ..model import IMAGINE_DIRECTIVE
from .rewards import RouteCost, answer_score, parse_route, tg_iou
from .routes import ASSISTANT_OPENER, FORCE_CLOSE, THINKS, RouteTokens, grid_route

# logit_bias is an ADD; a large finite value keeps the masked logits far below the rest without inf.
MASK_BIAS = -1.0e9
_TOKENS: dict[str, RouteTokens] = {}
_STOPS: dict[str, frozenset[int]] = {}


def route_tokens(tokenizer, model_path: str) -> RouteTokens:
    if model_path not in _TOKENS:
        with open(os.path.join(model_path, "config.json")) as fh:
            _TOKENS[model_path] = RouteTokens.from_pretrained(tokenizer, json.load(fh))
    return _TOKENS[model_path]


def _eos(tokenizer) -> list[int]:
    return [] if tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]


def stop_ids(tokenizer, model_path: str) -> frozenset[int]:
    """The tokens a generation stops on: the eos ids of ``generation_config.json`` (vLLM stops on them)
    and the tokenizer's eos."""
    if model_path not in _STOPS:
        path = os.path.join(model_path, "generation_config.json")
        eos = None
        if os.path.isfile(path):
            with open(path) as fh:
                eos = json.load(fh).get("eos_token_id")
        ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        _STOPS[model_path] = frozenset([*map(int, ids), *_eos(tokenizer)])
    return _STOPS[model_path]


def max_positions(model_path: str) -> int:
    """The model's context length, which verl's vLLM server takes when ``rollout.max_model_len`` is unset."""
    with open(os.path.join(model_path, "config.json")) as fh:
        config = json.load(fh)
    return int(config.get("max_position_embeddings") or config["text_config"]["max_position_embeddings"])


def placeholder_ids(processor) -> list[int]:
    """The processor's image and video placeholder ids (older transformers versions list them as
    ``image_ids`` / ``video_ids``, newer ones as ``image_token_ids`` / ``video_token_ids``)."""
    ids: set[int] = set()
    for modality in ("image", "video"):
        found = (getattr(processor, f"{modality}_token_ids", None) or getattr(processor, f"{modality}_ids", None)
                 or [getattr(processor, f"{modality}_token_id", None)])
        ids.update(int(i) for i in found if i is not None)
    return sorted(ids)


def empty_chain(tokenizer, ids: Sequence[int], think_close: int) -> bool:
    """Whether generated ``ids`` close the chain with nothing but whitespace before ``</think>``."""
    ids = list(ids)
    return think_close in ids and not tokenizer.decode(ids[:ids.index(think_close)]).strip()


def open_chain(ids: Sequence[int], inject: Sequence[int], think_close: int, stops: Collection[int]) -> bool:
    """Whether a chain is still open at its budget: ``</think>`` neither injected nor generated, and the
    generation not ended by a stop token."""
    return bool(ids) and ids[-1] not in stops and think_close not in ids and think_close not in inject


def add_directive(messages: list[dict]) -> list[dict]:
    """``messages`` with the imagine directive before the user's text, as the eval adds it."""
    messages = copy.deepcopy(messages)
    user = next(m for m in reversed(messages) if m["role"] == "user")
    if isinstance(user["content"], str):
        user["content"] = IMAGINE_DIRECTIVE + user["content"]
    else:
        text = next(part for part in user["content"] if part.get("type") == "text")
        text["text"] = IMAGINE_DIRECTIVE + text["text"]
    return messages


class ImagineRouteAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = self.config.avtr
        self.response_length = self.rollout_config.response_length
        self.group_size = self.rollout_config.n
        model_path = os.path.expanduser(self.config.actor_rollout_ref.model.path)
        self.max_model_len = self.rollout_config.max_model_len or max_positions(model_path)
        self.tokens = route_tokens(self.tokenizer, model_path)
        self.cost = RouteCost(self.stage.think_cost, self.stage.block_cost, self.stage.noblock_bonus,
                              self.stage.unclosed_min_chars)
        self.stops = stop_ids(self.tokenizer, model_path)
        # Out of the loss wherever they appear: the tokenizer's eos and the chat terminator, the tokens the
        # reference masks. A rollout that stops on another stop token (<|endoftext|> on Qwen3.5) trains it.
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.untrained = frozenset(i for i in (*_eos(self.tokenizer), im_end) if i is not None)
        self.close_ids = self.tokenizer.encode(FORCE_CLOSE, add_special_tokens=False)
        # The query rows of lm_head are zero, so the queries are banned too.
        self.banned = {i: MASK_BIAS for i in (*self.tokens.queries, *placeholder_ids(self.processor))}

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        task = kwargs["extra_info"]["task"]
        messages = list(kwargs["raw_prompt"])
        if self.stage.directive:
            messages = add_directive(messages)
        media = await self.process_multi_modal_info(messages)
        images, videos = media.get("images"), media.get("videos")
        mm_kwargs = self._get_mm_processor_kwargs()

        def render() -> tuple[str, list[int]]:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            text += ASSISTANT_OPENER
            inputs = build_multimodal_processor_inputs(self.processor, text=[text], images=images, videos=videos,
                                                       mm_processor_kwargs=mm_kwargs)
            return text, normalize_token_ids(inputs["input_ids"])

        text, prompt_ids = await self.loop.run_in_executor(None, render)
        engine_ids = self.tokenizer.encode(text, add_special_tokens=False)

        route = "visual" if task == "grounding" else grid_route(int(kwargs["session_id"]), self.group_size)
        inject = self.tokens.inject(route)
        budget = self.stage.grounding_max_tokens if task == "grounding" else self.stage.gate_max_tokens
        n = self.response_length
        params = dict(sampling_params, max_tokens=min(budget, n - len(inject.ids)), logit_bias=dict(self.banned))
        prompt = engine_ids + inject.ids
        # One sticky session for every call of the rollout, so they reuse the server's prefix cache (verl
        # gives each call its own vLLM request id).
        request_id = uuid4().hex
        calls = []

        async def generate(ids: list[int], **override):
            out = await self.server_manager.generate(request_id=request_id, prompt_ids=ids,
                                                     sampling_params={**params, **override}, image_data=images,
                                                     video_data=videos, mm_processor_kwargs=mm_kwargs)
            calls.append(out)
            return out

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            out = await generate(prompt)
            redraws, empty = 0, route in THINKS and empty_chain(self.tokenizer, out.token_ids, self.tokens.think_close)
            while empty and redraws < self.stage.empty_think_redraws:
                out = await generate(prompt)
                redraws, empty = redraws + 1, empty_chain(self.tokenizer, out.token_ids, self.tokens.think_close)
            chain = list(out.token_ids)
            chain_lps = list(out.log_probs or [0.0] * len(chain))
            close: list[int] = []
            answer: list[int] = []
            answer_lps: list[float] = []
            room = min(self.stage.force_close_tokens, n - len(inject.ids) - len(chain) - len(self.close_ids))
            # Skipped when the closed prompt would not fit the engine, which measures the expanded prompt.
            fits = len(prompt_ids) + len(inject.ids) + len(chain) + len(self.close_ids) < self.max_model_len
            if room > 0 and fits and open_chain(chain, inject.ids, self.tokens.think_close, self.stops):
                tail = await generate(prompt + chain + self.close_ids, max_tokens=room)
                close, answer = list(self.close_ids), list(tail.token_ids)
                answer_lps = list(tail.log_probs or [0.0] * len(answer))
        preempted = [c.num_preempted for c in calls if c.num_preempted is not None]
        metrics["num_preempted"] = sum(preempted) if preempted else -1
        fields = dict(calls[0].extra_fields)
        if "max_global_steps" in calls[-1].extra_fields:
            fields["max_global_steps"] = calls[-1].extra_fields["max_global_steps"]

        generated = chain + close + answer
        response_ids = inject.ids + generated
        # Injected tokens and the close are not the policy's samples; the chat terminator is not trained either.
        mask = [0] * len(inject.ids) + [1] * len(chain) + [0] * len(close) + [1] * len(answer)
        gates = list(inject.gates) if task == "gate" else []
        for offset in gates:
            mask[offset] = 1
        mask = [0 if t in self.untrained else m for t, m in zip(response_ids, mask, strict=True)]
        logprobs = [0.0] * len(inject.ids) + chain_lps + [0.0] * len(close) + answer_lps

        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        gold = kwargs["reward_model"]["ground_truth"]
        fired, closed, chars = parse_route(response)
        score = tg_iou(response, gold) if task == "grounding" else answer_score(response, gold)
        info = {"score": score, "fired": float(fired),
                "thinking": float(self.cost.thinking(closed, chars)), "think_chars": float(chars),
                "closed_think": float(closed), "truncated": float(not generated or generated[-1] not in self.stops),
                "resampled": float(redraws), "resample_exhausted": float(empty and redraws > 0),
                "force_closed": float(bool(close))}

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:n],
            response_mask=mask[:n],
            response_logprobs=logprobs[:n],
            multi_modal_data=media,
            mm_processor_kwargs=mm_kwargs,
            reward_score=score,
            num_turns=2,
            metrics=metrics,
            extra_fields={**fields, "reward_extra_info": info, "task": task, "route": route, "gates": gates,
                          "gate_kinds": [inject.gates[o] for o in gates], "turn_scores": [], "tool_rewards": []},
        )

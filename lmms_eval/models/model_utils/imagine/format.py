"""Prompt fragments shared by AVTR inference and evaluation.

The directive, query block, and reasoning separators must match training.
"""

from __future__ import annotations

import re

IMAGINE_OPEN = "<imagine>"
IMAGINE_CLOSE = "</imagine>"
# The vocabulary strings of the baked query tokens (the learnable queries that training feeds through
# a connector into the diffusion decoder), as the bake adds them to the tokenizer.
QUERY_TOKEN_FORMAT = "<latent_query_{}>"


def query_token(i: int, token_format: str = QUERY_TOKEN_FORMAT) -> str:
    """The tokenizer string of baked query token ``i``."""
    return token_format.format(i)

# Byte-identical to training's IMAGINE_DIRECTIVE, which every training sample carries. Any drift
# feeds the model a directive it has never seen while add_directive defaults True -- eval and
# training would silently disagree on the prompt.
IMAGINE_DIRECTIVE = (
    "If a visual clue would help you answer, generate it FIRST -- before you think -- as abstract "
    "visual tokens inside <imagine>...</imagine>, then think and answer.\n"
)
ASSISTANT_OPENER = "<|im_start|>assistant\n"

# logit_bias is an ADD (vllm/v1/sample/logits_processor/builtin.py), so a large negative is the
# id-precise equivalent of the HF path's suppress_tokens. Finite rather than -inf: it puts the term
# far under float32 resolution relative to the others, with no way for a downstream
# normalization to make a NaN.
MASK_BIAS = -1.0e9
# Forces a token (or a choice among a few) through logit_bias instead of allowed_token_ids. vLLM keeps
# allowed_token_ids in a per-slot mask that its persistent batch moves wrongly: swap_states swaps two
# row views (both rows end up with one request's mask) and condense never clears the vacated row
# (vllm/v1/worker/gpu_input_batch.py). The stale mask then applies to whatever request lands in the
# slot whenever any request in the batch uses allowed_token_ids, so a free-decoding answer next to a
# probe could emit nothing but "\n". logit_bias is a logits processor that tracks moves correctly. Raw
# logprobs are taken before either is applied, so the returned values are the same. FORCE_BIAS beats
# any logit, and fp32 still resolves the steps between the forced tokens' bf16 logits at that size.
FORCE_BIAS = 1.0e4
TOOL_CALL_TOKENS = ("<tool_call>", "</tool_call>")
# The turn terminators are <|im_end|> and <|endoftext|>. `until` only carries
# `tokenizer.eos_token`, which on this checkpoint is <|endoftext|> -- so the turn terminator the
# model actually emits, <|im_end|>, is NOT in it.
TURN_TERMINATORS = ("<|im_end|>", "<|endoftext|>")
# Everything render_scaffold emits before the reason/do-not-reason token.
IMAGINE_THINK_PREFIX = "<think>"
# An emitted block, in the baked rendering. Stripped before the task's extractor runs.
IMAGINE_BLOCK_RE: re.Pattern[str] = re.compile(
    r"<imagine>.*?(?:</imagine>|$)", re.DOTALL
)
THINK_FORCE_CLOSE = "\n</think>\n\n"
THINK_FORCE_ANSWER_MIN = 64


def render_block(
    num_queries: int, tail_newline: bool, token_format: str = QUERY_TOKEN_FORMAT
) -> str:
    """Render the query block with the same trailing separator used in training."""
    return (
        IMAGINE_OPEN
        + "".join(query_token(i, token_format) for i in range(num_queries))
        + IMAGINE_CLOSE
        + ("\n" if tail_newline else "")
    )


def render_scaffold(enable_thinking: bool) -> str:
    """Open reasoning, or prefill a CLOSED empty think block."""
    return "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"


def render_decline(tail_newline: bool) -> str:
    """Render the bare closing marker used by target-less training rows."""
    return IMAGINE_CLOSE + ("\n" if tail_newline else "")


def strip_terminators(text: str, terminators=TURN_TERMINATORS) -> str:
    """Remove turn terminators while retaining imagination markers for postprocessing."""
    for term in terminators:
        text = text.split(term)[0]
    return text


def as_bool(value: object, default: bool = False) -> bool:
    """``--model_args`` values arrive as str/int/bool depending on how they happen to parse."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

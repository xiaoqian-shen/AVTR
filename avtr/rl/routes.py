"""Token prefixes for direct, visual, textual, and combined reasoning routes."""

from __future__ import annotations

from dataclasses import dataclass

ROUTES = ("nothink", "visual", "text", "visual_text")
FIRES = frozenset({"visual", "visual_text"})
THINKS = frozenset({"text", "visual_text"})
GATE_THINK, GATE_VISUAL = 1, 2
# The prompt ends at a bare opener: the chat template's own generation prompt prefills <think>, which
# would put the first sampled token inside the think block.
ASSISTANT_OPENER = "<|im_start|>assistant\n"
# Appended to a chain still open at its budget, before the answer it is then given (the eval's
# THINK_FORCE_CLOSE).
FORCE_CLOSE = "\n</think>\n\n"


def grid_route(slot: int, group_size: int) -> str:
    """Route of rollout ``slot`` in a group: ``group_size / len(ROUTES)`` consecutive rollouts per route."""
    if group_size % len(ROUTES):
        raise ValueError(f"the route grid needs a group size divisible by {len(ROUTES)}, got {group_size}")
    return ROUTES[slot // (group_size // len(ROUTES))]


@dataclass(frozen=True)
class Injection:
    ids: list[int]
    gates: dict[int, int]  # offset in ``ids`` -> GATE_VISUAL / GATE_THINK


@dataclass(frozen=True)
class RouteTokens:
    imagine_open: int
    imagine_close: int
    queries: tuple[int, ...]
    think_open: int
    think_close: int
    newline: int
    double_newline: int

    @classmethod
    def from_pretrained(cls, tokenizer, config: dict) -> RouteTokens:
        """Ids of a ``bake --emit-markers`` checkpoint (its ``config.json`` and tokenizer)."""
        if not config.get("imagine_markers_emittable"):
            raise ValueError("RL needs a checkpoint baked with --emit-markers: the visual gate is sampled from the "
                             "marker rows of lm_head")
        start, n = int(config["latent_query_start_token_id"]), int(config["num_latent_queries"])

        def one(text: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{text!r} is not one token: {ids}")
            return ids[0]

        tokens = cls(int(config["imagine_start_token_id"]), int(config["imagine_end_token_id"]),
                     tuple(range(start, start + n)), one("<think>"), one("</think>"), one("\n"), one("\n\n"))
        for route in ROUTES:  # the ids must be what the eval's text scaffold tokenizes to
            if tokenizer.encode(tokens.render(route, tokenizer), add_special_tokens=False) != tokens.inject(route).ids:
                raise ValueError(f"the {route} scaffold does not round-trip through the tokenizer")
        return tokens

    def inject(self, route: str) -> Injection:
        fire = [self.imagine_open, *self.queries, self.imagine_close] if route in FIRES else [self.imagine_close]
        think = [self.newline] if route in THINKS else [self.double_newline, self.think_close, self.double_newline]
        return Injection(fire + [self.newline, self.think_open] + think, {0: GATE_VISUAL, len(fire) + 2: GATE_THINK})

    def render(self, route: str, tokenizer) -> str:
        return tokenizer.decode(self.inject(route).ids, skip_special_tokens=False)

    def gate_pair(self, kind: int) -> tuple[int, int]:
        """The two tokens a gate chooses between, (take, skip)."""
        return (self.imagine_open, self.imagine_close) if kind == GATE_VISUAL else (self.newline, self.double_newline)

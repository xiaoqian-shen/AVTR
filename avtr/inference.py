"""Single-question AVTR inference using the evaluation generation pipeline."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence


ROUTES = {
    "adaptive": dict(force_imagine_block=False, mute_imagine=False, enable_thinking=False,
                     probe_think=True, think_gate=True),
    "direct": dict(force_imagine_block=False, mute_imagine=True, enable_thinking=False,
                   probe_think=False, think_gate=False),
    "visual": dict(force_imagine_block=True, mute_imagine=False, enable_thinking=False,
                   probe_think=False, think_gate=False),
    "text": dict(force_imagine_block=False, mute_imagine=True, enable_thinking=True,
                 probe_think=False, think_gate=False),
    "visual_text": dict(force_imagine_block=True, mute_imagine=False, enable_thinking=True,
                        probe_think=False, think_gate=False),
}


def _checkpoint(model_name_or_path: str, revision: str | None) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.is_dir():
        return str(path.resolve())

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"],
    )


class AVTR:
    """Adaptive inference on a local or Hub-hosted AVTR checkpoint."""

    def __init__(self, backend: Any, max_new_tokens: int = 8192) -> None:
        self._backend = backend
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        route: str = "adaptive",
        revision: str | None = None,
        max_new_tokens: int = 8192,
        **kwargs: Any,
    ) -> AVTR:
        """Load a vocab-baked checkpoint; extra options configure the vLLM backend."""
        if route not in ROUTES:
            raise ValueError(f"Unknown route {route!r}; choose from {', '.join(ROUTES)}")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        checkpoint = _checkpoint(model_name_or_path, revision)
        with open(Path(checkpoint) / "config.json") as f:
            config = json.load(f)
        required = {"imagine_start_token_id", "imagine_end_token_id",
                    "latent_query_start_token_id", "num_latent_queries"}
        if not required.issubset(config):
            raise ValueError("Use an AVTR checkpoint exported with python -m avtr.tools.export --emit-markers")
        if route == "adaptive" and not config.get("imagine_markers_emittable"):
            raise ValueError("Adaptive routing requires a checkpoint exported with --emit-markers")

        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
        from lmms_eval.models.qwen3_5_imagine_vllm import Qwen3_5_ImagineVLLM

        options = dict(
            image_min_pixels=256 * 32 * 32,
            image_max_pixels=16384 * 32 * 32,
            video_min_pixels=16 * 32 * 32,
            video_max_pixels=256 * 32 * 32,
            video_total_pixels=25690112,
            max_frames=512,
            fps=2.0,
            max_new_tokens=max_new_tokens,
            think_budget=max_new_tokens,
        )
        options.update(kwargs)
        options.update(ROUTES[route])
        backend = Qwen3_5_ImagineVLLM(pretrained=checkpoint, **options)
        return cls(backend, max_new_tokens)

    def generate(
        self,
        question: str,
        *,
        images: Sequence[str | Path] | str | Path | None = None,
        video: str | Path | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Answer one question with ordered images, one video, or text alone."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if images is not None and video is not None:
            raise ValueError("Pass images or one video per question")
        budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if budget <= 0:
            raise ValueError("max_new_tokens must be positive")

        from lmms_eval.api.instance import Instance

        with ExitStack() as stack:
            visuals = []
            if images is not None:
                from PIL import Image

                paths = [images] if isinstance(images, (str, Path)) else images
                for path in paths:
                    source = stack.enter_context(Image.open(Path(path).expanduser()))
                    image = source.convert("RGB")
                    stack.callback(image.close)
                    visuals.append(image)
            if video is not None:
                path = Path(video).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.suffix not in {".mp4", ".avi", ".mov"}:
                    raise ValueError("Video inputs must use .mp4, .avi, or .mov")
                visuals.append(str(path))

            task, split = "avtr_inference", "input"
            self._backend.task_dict = {task: {split: [{}]}}
            request = Instance(
                request_type="generate_until",
                arguments=(question.strip(), {"max_new_tokens": budget}, lambda _: visuals,
                           0, task, split),
                idx=0,
                metadata={"task": task, "doc_id": 0, "repeats": 1},
            )
            return self._backend.generate_until([request])[0]


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _media_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local AVTR checkpoint or Hugging Face model ID")
    parser.add_argument("--revision", help="Hub branch, tag, or commit")
    parser.add_argument("--question", required=True)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--images", nargs="+", type=_media_path, help="Image files in prompt order")
    media.add_argument("--video", type=_media_path, help="One .mp4, .avi, or .mov file")
    parser.add_argument("--route", choices=ROUTES, default="adaptive")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
    parser.add_argument("--max-model-len", type=_positive_int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("--question must not be empty")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.video and Path(args.video).suffix not in {".mp4", ".avi", ".mov"}:
        parser.error("--video must use .mp4, .avi, or .mov")

    model = AVTR.from_pretrained(
        args.model,
        route=args.route,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(model.generate(args.question, images=args.images, video=args.video))


if __name__ == "__main__":
    main()

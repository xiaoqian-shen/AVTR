import base64
import json
import os
import re
import tempfile
import time
from io import BytesIO
from typing import Optional, Union

import fsspec
import torch
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.vision_process import (
    process_vision_info,
)
from lmms_eval.models.model_utils.qwen3_5_gen import (
    apply_bare_think,
    has_reasoning,
    majority_vote,
    postprocess_answer,
    resolve_attn_implementation,
    resolve_gen_kwargs,
    self_consistency_kwargs,
)
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration


@register_model("qwen3_5")
class Qwen3_5(lmms):
    """
    Qwen3.5 Model
    "https://huggingface.co/Qwen/Qwen3.5-4B"
    Qwen3.5 is a natively multimodal model supporting text, image,
    and video understanding. It uses Qwen3_5ForConditionalGeneration for
    loading and supports both dense variants.
    """

    # Opt-in generation options, all off. __init__ sets them per instance from its arguments,
    # so these class values only reach subclasses that skip __init__ (Qwen3_5VLLM overrides
    # them there).
    _bare_think: bool = False
    _raw_dump_path: Optional[str] = None
    _floor_max_new_tokens: bool = False
    _extract_final_answer: bool = False
    min_new_tokens: Optional[int] = None
    sample_temperature: Optional[float] = None
    sc_k: int = 1
    sc_temperature: float = 0.7

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        image_min_pixels: Optional[int] = 4 * 28 * 28,
        image_max_pixels: Optional[int] = 16384 * 28 * 28,
        video_min_pixels: Optional[int] = 128 * 28 * 28,
        video_max_pixels: Optional[int] = 768 * 28 * 28,
        video_total_pixels: Optional[int] = 115200 * 28 * 28,
        min_frames: Optional[int] = 4,
        max_frames: Optional[int] = 768,
        nframes: Optional[int] = None,
        fps: Optional[float] = 2.0,
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        bare_think: Optional[bool] = False,
        min_new_tokens: Optional[int] = None,
        sample_temperature: Optional[float] = None,
        raw_dump_path: Optional[str] = None,
        sc_k: Optional[int] = 1,
        sc_temperature: Optional[float] = 0.7,
        floor_max_new_tokens: Optional[bool] = False,
        extract_final_answer: Optional[bool] = False,
        left_padding: Optional[bool] = False,
        attn_fallback: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Initialize the Qwen3_5 model.

        Args:
            video_min_pixels (int, optional): The minimum pixels number for a frame of a video.
            video_max_pixels (int, optional): The maximum pixels number for a frame of a video.
            video_total_pixels (int, optional): The total pixels number for a video.
            min_frames (int, optional): The minimum frames number allowed for a video.
            max_frames (int, optional): The maximum frames number allowed for a video.
            nframes (int, optional): The exact number of frames to extract for a video. None means the frame
                number is decided by fps and max_frames.
            fps (float, optional): The fps to extract frames for a video.

        The remaining options default to the original behaviour; the imagine evals set them.
            bare_think (bool, optional): With enable_thinking, end the prefill at "<think>" instead of the
                template's "<think>\\n".
            min_new_tokens (int, optional): Force at least this many generated tokens (a task's own
                min_new_tokens wins). None disables it.
            sample_temperature (float, optional): Replace the task's temperature. None honors the task.
            raw_dump_path (str, optional): Directory that receives each rank's raw generations, with
                diagnostics, as JSONL.
            sc_k (int, optional): Sample this many answers per prompt and score the majority vote. A single
                sample disables it.
            sc_temperature (float, optional): Temperature for self-consistency sampling when the task decodes greedily.
            floor_max_new_tokens (bool, optional): Raise max_new_tokens to a floor, higher with thinking than without.
            extract_final_answer (bool, optional): Reduce the answer to its last <answer>...</answer> or
                \\boxed{...} content.
            left_padding (bool, optional): Pad batched prompts on the left. Batched generation needs it for
                correct generation and for the raw dump's think-decision probe.
            attn_fallback (bool, optional): With use_flash_attention_2, load with sdpa instead of failing when
                flash_attn does not import.
        """
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        if use_flash_attention_2:
            attn_impl = resolve_attn_implementation(fallback=bool(attn_fallback))
            if attn_impl != "flash_attention_2":
                eval_logger.warning(
                    f"flash_attn is not importable; loading {pretrained} with "
                    f"attn_implementation={attn_impl}"
                )
            self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation=attn_impl,
            ).eval()
        else:
            self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype="auto",
                device_map=self.device_map,
            ).eval()

        # data configuration for fetch_image
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        eval_logger.info(
            f"image_min_pixels: {self.image_min_pixels}, "
            f"image_max_pixels: {self.image_max_pixels}"
        )

        # data configuration for fetch_video
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.nframes = nframes
        self.fps = fps
        eval_logger.info(
            f"video_min_pixels: {self.video_min_pixels}, "
            f"video_max_pixels: {self.video_max_pixels}, "
            f"video_total_pixels: {self.video_total_pixels}, "
            f"min_frames: {self.min_frames}, "
            f"max_frames: {self.max_frames}, "
            f"nframes: {self.nframes}, "
            f"fps: {self.fps}"
        )

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        if left_padding:
            # Right padding leaves pad tokens between each shorter prompt and its continuation;
            # a batch of one is never padded.
            self.processor.tokenizer.padding_side = "left"
            self._tokenizer.padding_side = "left"
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = enable_thinking
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        self._bare_think = bool(bare_think)
        self.min_new_tokens = None if min_new_tokens is None else int(min_new_tokens)
        self.sample_temperature = (
            None if sample_temperature is None else float(sample_temperature)
        )
        self._raw_dump_path = raw_dump_path
        self.sc_k = max(1, int(sc_k or 1))
        self.sc_temperature = float(sc_temperature or 0.7)
        self._floor_max_new_tokens = bool(floor_max_new_tokens)
        self._extract_final_answer = bool(extract_final_answer)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_5")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_visual_dict(self, visual):
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            visual_dict = {
                "type": "video",
                "video": visual,
                "min_pixels": self.video_min_pixels,
                "max_pixels": self.video_max_pixels,
                "total_pixels": self.video_total_pixels,
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
                "fps": self.fps,
            }
            if self.nframes is not None:
                visual_dict["nframes"] = self.nframes
                visual_dict.pop("fps")
            return visual_dict
        elif isinstance(visual, Image.Image):
            base64_image = visual.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            return {
                "type": "image",
                "image": f"data:image/jpeg;base64,{base64_string}",
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
            }
        return None

    def _process_visuals(self, visuals):
        processed = []
        for visual in visuals:
            visual_dict = self._build_visual_dict(visual)
            if visual_dict is not None:
                processed.append(visual_dict)
        return processed

    def _build_interleaved_content(self, processed_visuals, context):
        content_parts = []

        if processed_visuals and processed_visuals[0]["type"] == "video":
            content_parts.append(processed_visuals[0])
            processed_visuals = processed_visuals[1:]

        image_placeholders = re.findall(r"<image \d+>", context)
        text_parts = re.split(r"<image \d+>", context)
        if text_parts[0]:
            content_parts.append({"type": "text", "text": text_parts[0]})

        for i, placeholder in enumerate(image_placeholders):
            img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
            image_idx = (
                min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
            )
            if processed_visuals and image_idx < len(processed_visuals):
                content_parts.append(processed_visuals[image_idx])
            if i + 1 < len(text_parts) and text_parts[i + 1]:
                content_parts.append({"type": "text", "text": text_parts[i + 1]})

        return content_parts

    def _build_messages(self, contexts, visual_list, interleave_visuals):
        batched_messages = []
        for i, context in enumerate(contexts):
            processed_visuals = self._process_visuals(visual_list[i])

            if not interleave_visuals:
                content = processed_visuals + [{"type": "text", "text": context}]
            else:
                content = self._build_interleaved_content(processed_visuals, context)

            batched_messages.append([{"role": "user", "content": content}])
        return batched_messages

    def _render_texts(self, batched_messages):
        """The prompt string of each message list. Subclasses that prefill their own assistant
        prefix (the imagine adapters) override this."""
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for msg in batched_messages
        ]
        return apply_bare_think(texts, self._bare_think, self.enable_thinking)

    def _prepare_inputs(self, batched_messages):
        texts = self._render_texts(batched_messages)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batched_messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = (
                list(video_inputs),
                list(video_metadatas),
            )
        else:
            video_metadatas = None

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            do_resize=False,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)
        return inputs

    def _resolve_gen_kwargs(self, gen_kwargs):
        return resolve_gen_kwargs(
            gen_kwargs,
            enable_thinking=self.enable_thinking,
            floor_max_new_tokens=self._floor_max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            sample_temperature=self.sample_temperature,
        )

    def _resolve_until(self, gen_kwargs):
        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(
                f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
            )
        # Avoid using '\n\n' as a stopper for Qwen3_5 to prevent truncation, which can lead to incorrect results
        return [item for item in until if item != "\n\n"]

    def _postprocess_answer(self, ans, until):
        return postprocess_answer(
            ans,
            until,
            enable_thinking=self.enable_thinking,
            extract_final=self._extract_final_answer,
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        raw_records: list[dict] = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temperature sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = self._resolve_until(gen_kwargs)

            contexts = [c.replace("<image>", "") for c in contexts]

            interleave_visuals = self.interleave_visuals
            if task == "video_mmmu_adaptation":
                interleave_visuals = True
                eval_logger.info("Interleaving visuals for video_mmmu_adaptation")

            batched_messages = self._build_messages(
                contexts, visual_list, interleave_visuals
            )
            inputs = self._prepare_inputs(batched_messages)

            current_gen_kwargs = self._resolve_gen_kwargs(gen_kwargs)
            answers, dump_info = self._generate(
                inputs, current_gen_kwargs, len(contexts)
            )

            k = self.sc_k
            for i, (context, did) in enumerate(zip(contexts, doc_id)):
                raws = answers[i * k : (i + 1) * k]
                cands = [self._postprocess_answer(r, until) for r in raws]
                ans = majority_vote(cands)
                if dump_info is not None:
                    raw_records.append(
                        self._raw_record(dump_info, i, did, task, raws[0], cands, ans)
                    )
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        self._maybe_dump_raw(raw_records)
        return res

    def _generate(self, inputs, gen_kwargs, n_prompts):
        """Decode one batch into ``sc_k`` answers per prompt, prompt-major.

        Also returns the batch diagnostics the raw dump records, or None without raw_dump_path.
        """
        gen_kwargs, extra = self_consistency_kwargs(
            gen_kwargs, self.sc_k, self.sc_temperature
        )
        if gen_kwargs.get("min_new_tokens") is not None:
            extra["min_new_tokens"] = gen_kwargs["min_new_tokens"]
        want_dump = bool(self._raw_dump_path)
        if want_dump:
            # Per-step scores give the first generated token's distribution: for a no-think MCQ
            # answer that token IS the letter, so its confidence comes free with the prefill.
            extra["output_scores"] = True
            extra["return_dict_in_generate"] = True

        pad_token_id = self.tokenizer.pad_token_id
        t_gen = time.perf_counter()
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=gen_kwargs["do_sample"],
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            top_k=gen_kwargs["top_k"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            **extra,
        )
        elapsed = time.perf_counter() - t_gen
        info = None
        if want_dump:
            info = {"first_conf": self._first_token_conf(cont, n_prompts, self.sc_k)}
            cont = cont.sequences

        # Every row of a padded batch has the prompt's padded length, so one slice point also
        # serves the k-expanded rows.
        n_prompt = int(inputs.input_ids.shape[1])
        generated_ids_trimmed = [out_ids[n_prompt:] for out_ids in cont]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if info is not None:
            info["decision"] = (
                self._think_decision_probs(inputs)
                if self.enable_thinking
                else [{}] * n_prompts
            )
            info["n_prompt"] = n_prompt
            info["n_gen"] = [
                int((row != pad_token_id).sum()) for row in generated_ids_trimmed
            ]
            info["batch_gen_s"] = round(elapsed, 3)
            info["batch_rows"] = len(cont)
        return answers, info

    def _raw_record(self, info, i, doc_id, task, raw, cands, ans) -> dict:
        """Raw-dump row for prompt ``i`` of a batch; ``raw`` keeps the think block that scoring
        strips."""
        k = self.sc_k
        rec = {
            "doc_id": int(doc_id),
            "task": task,
            "has_think": has_reasoning(raw),
            "decision": info["decision"][i],
            "first_conf": info["first_conf"][i],
            "raw": raw[:8000],
            "answer": ans[:400],
            "n_prompt": info["n_prompt"],
            "n_gen": info["n_gen"][i * k : (i + 1) * k],
            "batch_gen_s": info["batch_gen_s"],
            "batch_rows": info["batch_rows"],
        }
        if k > 1:
            rec["cands"] = [c[:200] for c in cands]
        return rec

    def _maybe_dump_raw(self, raw_records: list[dict]) -> None:
        """Write this rank's raw generations to ``<raw_dump_path>/<task>_rank<r>.jsonl``, named
        after the first record's task. No-op unless raw_dump_path is set; never fails the eval."""
        if not self._raw_dump_path or not raw_records:
            return
        try:
            task = raw_records[0]["task"]
            local = os.path.join(
                tempfile.gettempdir(),
                f"qwen3_5_raw_{task}_rank{self.rank}.jsonl",
            )
            with open(local, "w") as f:
                for r in raw_records:
                    f.write(json.dumps(r) + "\n")
            out_dir = self._raw_dump_path.rstrip("/")
            dst = f"{out_dir}/{task}_rank{self.rank}.jsonl"
            fs, out_rel = fsspec.core.url_to_fs(out_dir)
            fs.makedirs(out_rel, exist_ok=True)
            fs.put_file(local, f"{out_rel}/{task}_rank{self.rank}.jsonl")
            eval_logger.info("qwen3_5: dumped {} raw gens -> {}", len(raw_records), dst)
        except Exception as e:  # noqa: BLE001 -- debug dump must never break eval
            eval_logger.warning("qwen3_5 raw dump failed: {}", e)

    def _first_token_conf(self, gen_out, n_prompts: int, k: int) -> list[dict]:
        """``p_top1``, margin, entropy and ``top5`` of the FIRST generated token, per prompt
        (row ``i * k`` of the k-expanded batch). Best-effort; ``{}`` per prompt on any error."""
        try:
            scores = getattr(gen_out, "scores", None)
            if not scores:
                return [{}] * n_prompts
            # [B*k, V], the pre-sampling distribution of the first token
            probs = torch.softmax(scores[0].float(), dim=-1)
            out: list[dict] = []
            for i in range(n_prompts):
                row = probs[i * k]
                top_p, top_i = torch.topk(row, 5)
                ent = float(-(row * row.clamp_min(1e-12).log()).sum())
                out.append(
                    {
                        "p_top1": round(float(top_p[0]), 5),
                        "margin": round(float(top_p[0] - top_p[1]), 5),
                        "entropy": round(ent, 4),
                        "top5": [
                            (self.tokenizer.decode([int(j)]), round(float(p), 4))
                            for p, j in zip(top_p, top_i)
                        ],
                    }
                )
            return out
        except Exception as e:  # noqa: BLE001 -- diagnostic must never break eval
            eval_logger.warning("first-token confidence probe failed: {}", e)
            return [{}] * n_prompts

    def _think_decision_probs(self, inputs) -> list[dict]:
        """Per batch row, the next-token distribution right after the prefilled ``<think>``
        opener (index -1 for every row under left padding): the probability of the empty
        (no-think) continuation versus starting real reasoning. One extra forward pass.
        Best-effort; ``{}`` per row on any error."""
        try:
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)

            def _single_id(s):
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                return ids[0] if len(ids) == 1 else None

            nl_id = _single_id("\n")
            nl2_id = _single_id("\n\n")
            et_id = _single_id("</think>")
            out: list[dict] = []
            for b in range(probs.shape[0]):
                row = probs[b]
                top_p, top_i = torch.topk(row, 5)
                ent = float(-(row * row.clamp_min(1e-12).log()).sum())
                out.append(
                    {
                        "p_newline": float(row[nl_id]) if nl_id is not None else None,
                        # Which id means "no think" flips with the prefill: after "<think>\n"
                        # it is a second "\n"; after a bare "<think>" it is "\n\n". Both are
                        # reported so the dump reads under either layout.
                        "p_newline2": float(row[nl2_id])
                        if nl2_id is not None
                        else None,
                        "p_no_think": float(row[nl2_id if self._bare_think else nl_id]),
                        "p_endthink": float(row[et_id]) if et_id is not None else None,
                        "entropy": round(ent, 3),
                        "top5": [
                            (self.tokenizer.decode([int(i)]), round(float(p), 4))
                            for p, i in zip(top_p, top_i)
                        ],
                    }
                )
            return out
        except Exception as e:  # noqa: BLE001 -- diagnostic must never break eval
            eval_logger.warning("think-decision probe failed: {}", e)
            n = int(inputs["input_ids"].shape[0]) if "input_ids" in inputs else 1
            return [{} for _ in range(n)]

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

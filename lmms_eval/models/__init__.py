import importlib
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    # Qwen3.5 through HF transformers; the base class of the vLLM backends.
    "qwen3_5": "Qwen3_5",
    # qwen3_5's prompts and answer handling, generated through vLLM.
    "qwen3_5_vllm": "Qwen3_5VLLM",
    # Adaptive imagine checkpoints; needs a vocab-baked dir from `scripts/bake.sh --emit-markers`.
    "qwen3_5_imagine_vllm": "Qwen3_5_ImagineVLLM",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"

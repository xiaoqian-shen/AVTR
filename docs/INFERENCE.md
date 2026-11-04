# Inference

The released AVTR Hugging Face checkpoints can be loaded directly. Latent queries and
visual-imagination markers are already stored in their vocabularies, so inference runs
the Qwen3.5 backbone without SANA.
The inference interface uses the same prompt construction and generation pipeline as evaluation.

## Load a model

```python
from avtr import AVTR

model = AVTR.from_pretrained("HF_CKPT_PATH")
answer = model.generate(
    "Which image shows the object from a different viewpoint?",
    images=["reference.jpg", "view_a.jpg", "view_b.jpg"],
)
print(answer)
```

`from_pretrained` accepts a local directory or a Hugging Face model repository ID.
For a Hub model, `revision=` selects a branch, tag, or commit. The
[AVTR collection](https://huggingface.co/collections/shenxq/avtr) is the model release location.

## Command line

```bash
python -m avtr.inference --model HF_CKPT_PATH \
    --images image.jpg --question "What is happening in this image?"

python -m avtr.inference --model HF_CKPT_PATH \
    --video video.mp4 --question "When does the person sit down? Give the interval in seconds."
```

Image paths follow their order in the prompt; video inputs support `.mp4`, `.avi`, and `.mov`
files and require FFmpeg.
Each call accepts images or one video.

## Reasoning routes

| Route | Visual imagination | Textual reasoning |
|---|---|---|
| `adaptive` (default) | Model decides | Model decides |
| `direct` | Disabled | Disabled |
| `visual` | Enabled | Disabled |
| `text` | Disabled | Enabled |
| `visual_text` | Enabled | Enabled |

```bash
python -m avtr.inference --model HF_CKPT_PATH --route visual \
    --images image.jpg --question "How many objects are visible?"
```

In Python, choose the route when loading: `AVTR.from_pretrained(path, route="visual")`.

## Generation settings

Greedy decoding and an 8192-token generation budget match the evaluation defaults.
The returned string contains the final answer, with the latent-query block and reasoning scaffold removed.

```python
model = AVTR.from_pretrained(
    "HF_CKPT_PATH",
    max_new_tokens=8192,
    max_model_len=40960,
    gpu_memory_utilization=0.85,
)
answer = model.generate("Describe this image.", images=["image.jpg"], max_new_tokens=2048)
```

The CLI exposes `--max-new-tokens`, `--max-model-len`, and `--gpu-memory-utilization`.
The backend runs on one GPU; select it with `CUDA_VISIBLE_DEVICES`.
For benchmark evaluation across GPUs, use [scripts/eval.sh](../scripts/eval.sh).

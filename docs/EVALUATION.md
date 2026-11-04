# Evaluation

The bundled [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness includes Qwen3.5
baseline adapters and AVTR adaptive inference for 16 benchmarks.

## Running

Use a released AVTR Hugging Face checkpoint:

```bash
# TASKS:
#   image: vstar hrbench_4k hrbench_8k mmsi_bench blink realworldqa erqa
#   video: mvbench videomme longvideobench_val_v mlvu vsibench
#   tg: charades_sta activitynet_tvg nextgqa qvhighlights
#   all: all benchmarks listed above
CKPT=HF_CKPT_PATH OUT=EVAL_OUTPUT_PATH TASKS=all scripts/eval.sh

# Selected tasks and routes
CKPT=HF_CKPT_PATH OUT=EVAL_OUTPUT_PATH TASKS="erqa vstar video" ARMS="adaptive force_think" scripts/eval.sh
```

| Variable | Purpose |
|---|---|
| `CKPT` | Local HF-format checkpoint directory |
| `TASKS` | Task names or `image` (default), `video`, `tg`, `all` |
| `ARMS` | `adaptive`, `force_think`, `force_nothink`; defaults to all three |
| `NUM_GPUS` | GPUs per run; defaults to all visible GPUs |
| `LIMIT` | Optional row limit per task |
| `OUT` | Output root; default `./eval_results` |
| `HF_TOKEN` | Access token for gated datasets |
| `HF_HOME` | Dataset and model cache directory |
| `EXTRA_MODEL_ARGS` | Additional adapter arguments, prefixed with a comma |

The three arms are:

- `adaptive`: the model decides whether to imagine and whether to reason.
- `force_think`: visual imagination and textual reasoning are always enabled.
- `force_nothink`: visual imagination is enabled and textual reasoning is disabled.

Temporal-grounding tasks use `force_nothink`.
Each GPU runs one vLLM engine; multi-GPU evaluation uses data parallelism.
Generation defaults to greedy decoding with an 8192-token budget.
Advanced options and pixel budgets are listed in [scripts/eval.sh](../scripts/eval.sh).

## Benchmarks and data

| Group | Tasks |
|---|---|
| `image` | `vstar`, `hrbench_4k`, `hrbench_8k`, `mmsi_bench`, `blink`, `realworldqa`, `erqa` |
| `video` | `mvbench`, `videomme`, `longvideobench_val_v`, `mlvu`, `vsibench` |
| `tg` | `charades_sta`, `activitynet_tvg`, `nextgqa`, `qvhighlights` |

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

Temporal-grounding tasks use `force_nothink` unless `ARMS` is explicitly set.
Each GPU runs one vLLM engine; multi-GPU evaluation uses data parallelism.
Generation defaults to greedy decoding with an 8192-token budget.
Advanced options and pixel budgets are listed in [scripts/eval.sh](../scripts/eval.sh).

## Benchmarks and data

| Group | Tasks |
|---|---|
| `image` | `vstar`, `hrbench_4k`, `hrbench_8k`, `mmsi_bench`, `blink`, `realworldqa`, `erqa` |
| `video` | `mvbench`, `videomme`, `longvideobench_val_v`, `mlvu`, `vsibench` |
| `tg` | `charades_sta`, `activitynet_tvg`, `nextgqa`, `qvhighlights` |

Datasets are downloaded from public sources and cached under `HF_HOME`.
Evaluation uses the full released evaluation split: val for BLINK, QVHighlights, and LongVideoBench;
val_2 for ActivityNet-TVG; dev for MLVU; test or the only released split for the others.
Video-MME is evaluated without subtitles.

BLINK and VSI-Bench are pinned to Hub commits; QVHighlights annotations are pinned to a commit
and SHA-256. Other Hub datasets use their current default revisions; MVBench uses its video branch.
A sample with missing media fails the run.

### Dataset-specific setup

- **LongVideoBench:** accept the terms for
  [longvideobench/LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench)
  and set `HF_TOKEN`.
- **RealWorldQA:** the harness uses `nirajandhakal/realworldqa`. Its question texts differ from
  `xai-org/RealworldQA` for 36 of 765 rows, so scores on the two versions are not directly comparable.
- **BLINK:** the harness reads the 14 validation subsets from
  [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK).
- **VSI-Bench:** the pinned revision contains the original, non-debiased evaluation files.
- **QVHighlights:** the first run streams the authors' approximately 144 GB archive and retains
  the 1519 validation videos, approximately 18 GB. This happens before evaluation even with `LIMIT`.
  Set `QVHIGHLIGHTS_VIDEO_DIR` to a directory of `<vid>.mp4` files to reuse local videos, or
  `QVHIGHLIGHTS_VIDEO_ARCHIVE` to a local archive or URL.

Existing extracted media directories can be reused under the dataset cache paths.
Video benchmarks require torchcodec and FFmpeg 4–7 shared libraries;
see [installation](INSTALLATION.md).

## Outputs

Results are written to `$OUT/<model>/<arm>/<task>/`.
The model name is the checkpoint directory name; `checkpoint-N` directories include their parent
name. Use a distinct `OUT` for each RL step, whose checkpoint directories are all named `huggingface`.

A failed task is recorded while the remaining tasks continue. The launcher exits with an error
if any task failed. Each task saves its benchmark scores in a `*_results.json` file alongside
per-sample logs.

Per-sample route decisions and confidence values are in `raw_dump/*.jsonl`.
The raw dump is cleared on each run. The harness also writes sample logs and results JSON files
under each task directory.

Temporal grounding reports R@1 at IoU 0.3, 0.5, and 0.7, plus mean IoU.
NExT-GQA also reports answer accuracy. QVHighlights official mAP is not computed.

For speed comparisons, use logged generation time rather than total runtime, which also includes model
loading, compilation, warmup, and scoring. Use the same host and comparable load conditions.

## Reproducibility

Deterministic compilation, fixed GatedDeltaNet kernel configurations, and pinned engine step ordering
are enabled by default. Keep the software build, GPU type, and GPU count fixed when comparing runs:
kernel selection and batching can change greedy outputs.

The imagine pipeline uses vLLM engine internals, so the vLLM version is pinned.
Compilation caches are isolated per evaluation process and removed on exit; set
`AVTR_COMPILE_CACHE_ROOT` to choose their temporary storage location.

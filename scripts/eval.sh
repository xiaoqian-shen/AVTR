#!/bin/bash
# Usage: CKPT=<HF checkpoint> TASKS=<tasks> ARMS=<arms> scripts/eval.sh
# Benchmark setup, outputs, and reproducibility: docs/EVALUATION.md.
#
# TASKS: image (default), video, tg, all, or individual task names.
# ARMS: adaptive, force_think, force_nothink; TG defaults to force_nothink.
# Common env: OUT, NUM_GPUS, LIMIT, NAME_TAG, PYTHON, MAIN_PORT, HF_TOKEN.
# Media: QVHIGHLIGHTS_VIDEO_DIR, QVHIGHLIGHTS_VIDEO_ARCHIVE, FORCE_QWENVL_VIDEO_READER.
# Decoding: GEN_MAX_NEW, THINK_BUDGET, TEMPERATURE, PROBE_MODE.
# Routing: THINK_THRESHOLD, THRESHOLD, GATE, NUM_QUERIES, ADD_DIRECTIVE,
#          BLOCK_TAIL_NL, EMIT_DECLINE, SINGLE_BLOCK, DUMP_ANSWER_CONF.
# Budgets: IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS,
#          VIDEO_TOTAL_PIXELS, VIDEO_MAX_FRAMES, VIDEO_FPS, PRESET=tg.
# Diagnostics and overrides: RAW_DUMP, EXTRA_MODEL_ARGS.

set -euo pipefail

: "${CKPT:?set CKPT to an HF-format AVTR checkpoint directory}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT%/}"
if ! grep -q '"latent_query_start_token_id"' "${CKPT}/config.json" 2>/dev/null; then
    echo "${CKPT}/config.json has no latent_query_start_token_id: use an AVTR Hugging Face checkpoint." \
         "See docs/TRAINING.md to export a training checkpoint." >&2
    exit 2
fi
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null; then
    echo "${PYTHON} not found: activate the eval environment or set PYTHON." >&2
    exit 2
fi

IMAGE_TASKS="vstar hrbench_4k hrbench_8k mmsi_bench blink realworldqa erqa"
VIDEO_TASKS="mvbench videomme longvideobench_val_v mlvu vsibench"
TG_TASKS="charades_sta activitynet_tvg nextgqa qvhighlights"
tasks=()
for t in ${TASKS:-image}; do
    case "${t}" in
        image) tasks+=(${IMAGE_TASKS}) ;;
        video) tasks+=(${VIDEO_TASKS}) ;;
        tg)    tasks+=(${TG_TASKS}) ;;
        all)   tasks+=(${IMAGE_TASKS} ${VIDEO_TASKS} ${TG_TASKS}) ;;
        *)     tasks+=("${t}") ;;
    esac
done
read -ra arms <<< "${ARMS:-adaptive force_think force_nothink}"
for arm in "${arms[@]}"; do
    case "${arm}" in
        adaptive|force_think|force_nothink) ;;
        *) echo "unknown arm '${arm}' (adaptive|force_think|force_nothink)" >&2; exit 2 ;;
    esac
done

if [ "${#tasks[@]}" -eq 0 ] || [ "${#arms[@]}" -eq 0 ]; then
    echo "nothing to run (TASKS='${TASKS:-image}', ARMS='${ARMS:-}')" >&2
    exit 2
fi

# PRESET=tg: an alternative temporal-grounding budget (fewer, larger frames). Without
# it every task, TG included, uses the defaults below.
if [ "${PRESET:-}" = "tg" ]; then
    VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
    VIDEO_FPS="${VIDEO_FPS:-2}"
    VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-$((20000 * 32 * 32))}"
    VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((512 * 32 * 32))}"
fi

# Visual budgets, in tokens times the pixels of one Qwen3.5 token (patch size x spatial merge, per side).
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-$((256 * 32 * 32))}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-$((16384 * 32 * 32))}"
# A cap, not a target: frames = min(duration x VIDEO_FPS, VIDEO_MAX_FRAMES).
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-512}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-25690112}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((16 * 32 * 32))}"
VIDEO_FPS="${VIDEO_FPS:-2}"
# Overrides every task's own budget (the tasks' short budgets would truncate the imagine block and any chain).
GEN_MAX_NEW="${GEN_MAX_NEW:-8192}"

ckpt_dir="$(cd "${CKPT}" && pwd)"
model_name="$(basename "${ckpt_dir}")"
if [[ "${model_name}" == checkpoint-* ]]; then
    model_name="$(basename "$(dirname "${ckpt_dir}")")-${model_name}"
fi
OUT="${OUT:-${PWD}/eval_results}"
OUT="${OUT%/}/${model_name}"

if [ -n "${NUM_GPUS:-}" ]; then
    ngpu="${NUM_GPUS}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _devs <<< "${CUDA_VISIBLE_DEVICES}"
    ngpu="${#_devs[@]}"
else
    ngpu="$(nvidia-smi -L | wc -l)"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# torchcodec loads the FFmpeg shared libraries when imported. A failed load does not fail the run:
# every video falls back to torchvision's reader (a warning each), which decodes whole clips and
# samples other frames. So the import is checked once, up front.
video_tasks=()
for t in "${tasks[@]}"; do
    if [[ " ${VIDEO_TASKS} ${TG_TASKS} " == *" ${t} "* || "${t}" == mvbench_* ]]; then
        video_tasks+=("${t}")
    fi
done
if [ "${FORCE_QWENVL_VIDEO_READER}" = torchcodec ] && [ "${#video_tasks[@]}" -gt 0 ] \
        && ! tc_err="$("${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder' 2>&1)"; then
    echo "FORCE_QWENVL_VIDEO_READER=torchcodec reads the videos of ${video_tasks[*]}, but" \
         "'${PYTHON} -c \"from torchcodec.decoders import VideoDecoder\"' fails:" >&2
    grep -m 1 -E '^[A-Za-z_.]*(Error|Exception)' <<< "${tc_err}" >&2 \
        || tail -n 1 <<< "${tc_err}" >&2
    echo "Install the FFmpeg 4-7 shared libraries (e.g. apt install ffmpeg, or" \
         "conda install -c conda-forge \"ffmpeg<8\") where ${PYTHON} can load them." >&2
    exit 2
fi

if [ "${ngpu}" -gt 1 ]; then
    # accelerate's launcher, run by ${PYTHON}. Without --multi_gpu / --gpu_ids, an accelerate
    # default config ($HF_HOME/accelerate/default_config.yaml) decides the distributed type and the
    # GPUs, e.g. a single process on the first GPU whatever NUM_GPUS says.
    launcher=("${PYTHON}" -m accelerate.commands.launch --multi_gpu --gpu_ids all
              --num_processes "${ngpu}" --num_machines 1 --mixed_precision no --dynamo_backend no
              --main_process_port "${MAIN_PORT:-29511}" -m lmms_eval)
else
    launcher=("${PYTHON}" -m lmms_eval)
fi

# Ctrl-C ends the sweep. Without the trap, a launcher that exits normally on SIGINT (accelerate's
# does) only fails the current run, and the loop goes on with the next one.
trap 'exit 130' INT

failed=()
for arm in "${arms[@]}"; do
    case "${arm}" in
        adaptive)      force=False; think=False; probe=True;  think_gate=True ;;
        force_think)   force=True;  think=True;  probe=False; think_gate=False ;;
        force_nothink) force=True;  think=False; probe=False; think_gate=False ;;
    esac
    arm_out="${OUT}/${arm}${NAME_TAG:-}"

    model_args="pretrained=${CKPT}/,num_queries=${NUM_QUERIES:-64},max_new_tokens=${GEN_MAX_NEW}"
    model_args+=",image_min_pixels=${IMAGE_MIN_PIXELS},image_max_pixels=${IMAGE_MAX_PIXELS}"
    model_args+=",max_frames=${VIDEO_MAX_FRAMES},video_total_pixels=${VIDEO_TOTAL_PIXELS}"
    model_args+=",video_min_pixels=${VIDEO_MIN_PIXELS},video_max_pixels=${VIDEO_MAX_PIXELS}"
    model_args+=",fps=${VIDEO_FPS},enable_thinking=${think}"
    model_args+=",add_directive=${ADD_DIRECTIVE:-True},block_tail_newline=${BLOCK_TAIL_NL:-True}"
    model_args+=",emit_decline=${EMIT_DECLINE:-True},single_block=${SINGLE_BLOCK:-True}"
    model_args+=",force_imagine_block=${force},mute_imagine=False"
    model_args+=",imagine_prob_threshold=${THRESHOLD:-none},imagine_gate=${GATE:-think}"
    model_args+=",probe_think=${probe},think_gate=${think_gate},think_threshold=${THINK_THRESHOLD:-None}"
    # A chain still open after THINK_BUDGET tokens is closed with "</think>" and answered in one more
    # round, instead of being scored on its truncated text. No effect on the no-think arms.
    model_args+=",think_budget=${THINK_BUDGET:-${GEN_MAX_NEW}}"
    model_args+=",dump_answer_conf=${DUMP_ANSWER_CONF:-False}"
    # The model's own sampling setting; it never reads the task's temperature.
    model_args+=",temperature=${TEMPERATURE:-0.0}"
    [ -z "${PROBE_MODE:-}" ] || model_args+=",probe_mode=${PROBE_MODE}"

    for task in "${tasks[@]}"; do
        if [ -z "${ARMS:-}" ] && [[ " ${TG_TASKS} " == *" ${task} "* ]] && [ "${arm}" != force_nothink ]; then
            continue
        fi
        task_out="${arm_out}/${task}"
        task_args="${model_args}"
        # The fire and think rates live only in the per-sample dump (field `fired`, `think_decided`).
        [ "${RAW_DUMP:-1}" = "0" ] || task_args+=",raw_dump_path=${task_out}/raw_dump"
        task_args+="${EXTRA_MODEL_ARGS:-}"
        # Remove stale records from earlier runs before writing the current per-rank dumps.
        rm -rf "${task_out}/raw_dump"
        mkdir -p "${task_out}"
        echo "== ${model_name} arm=${arm} task=${task} gpus=${ngpu} -> ${task_out}"
        task_cli_args=(--tasks "${task}")
        if [ -n "${LIMIT:-}" ]; then
            task_cli_args+=(--limit "${LIMIT}")
        fi
        rc=0
        "${launcher[@]}" \
            --model qwen3_5_imagine_vllm \
            --model_args "${task_args}" \
            --gen_kwargs "max_new_tokens=${GEN_MAX_NEW}" \
            "${task_cli_args[@]}" \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${arm}" \
            --output_path "${task_out}" \
            --fail_on_error || rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "== FAILED (exit ${rc}): ${model_name} arm=${arm} task=${task}; continuing" >&2
            failed+=("${arm}/${task}")
        fi
    done
done
echo "Results -> ${OUT}/<arm>/<task>/"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "${#failed[@]} run(s) failed: ${failed[*]}" >&2
    exit 1
fi
exit 0

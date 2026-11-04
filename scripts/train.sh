#!/bin/bash
# Usage: scripts/train.sh <0|1|2> [key.path=value ...]
# Data, checkpoint, and distributed settings: docs/TRAINING.md.
set -euo pipefail

STAGE="${1:?usage: $0 <0|1|2> [key.path=value ...]}"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"

exec torchrun \
    --nproc_per_node="${NPROC:-8}" \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m avtr.train --config "${REPO}/scripts/configs/stage${STAGE}.yaml" "$@"

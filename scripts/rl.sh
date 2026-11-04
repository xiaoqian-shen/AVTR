#!/bin/bash
# Usage: CKPT=<baked checkpoint> RL_DATA=<train.parquet> scripts/rl.sh [key.path=value ...]
# Data formats and Ray cluster setup: docs/TRAINING.md#adaptive-reasoning-with-rl.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?set CKPT to a baked checkpoint dir}"
: "${RL_DATA:?set RL_DATA to the parquet written by avtr/rl/data.py}"
export AVTR_REPO="${REPO}" OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/runs/rl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"

exec python -m avtr.rl.train --config-path "${REPO}/scripts/configs" --config-name rl "$@"

#!/bin/bash
# Usage: CKPT=<stage-2 checkpoint> OUT=<export directory> scripts/bake.sh [--emit-markers]
# Local and fsspec checkpoint export: docs/TRAINING.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m avtr.tools.bake --ckpt "${CKPT:?set CKPT}" --out "${OUT:?set OUT}" --work "${WORK:-/tmp/avtr_bake}" "$@"

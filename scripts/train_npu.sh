#!/usr/bin/env bash
# Single-card Ascend 910B Flash3D pre-training launcher.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script was not found: ${CANN_ENV}" >&2
  exit 1
fi
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "${PROJECT_ROOT}"
python scripts/check_npu_env.py
# EXPERIMENT selects the NPU pre-training profile:
#   layered_re10k_npu       (default) DA V1 ViT-B, frozen
#   layered_re10k_npu_vits            DA V1 ViT-S, frozen (lower memory)
EXPERIMENT="${EXPERIMENT:-layered_re10k_npu}"
exec python train.py "+experiment=${EXPERIMENT}" "$@"

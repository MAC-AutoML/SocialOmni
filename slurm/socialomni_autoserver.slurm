#!/bin/bash
#SBATCH --job-name=socialomni
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p "slurm/logs" "results"

# 可通过 sbatch --export 覆盖：
#   MODEL=qwen3_omni / qwen3_omni_thinking / qwen2_5_omni / miniomni_2 / omnivinci
#   LEVEL=1 或 2
#   MAX_SAMPLES 为空表示跑完整任务；设为 1/10 等用于冒烟测试
LEVEL="${LEVEL:-1}"
MODEL="${MODEL:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-1}"
REUSE_SERVER="${REUSE_SERVER:-0}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
TEST_TIMEOUT="${TEST_TIMEOUT:-86400}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

GPU_COUNT_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
GPU_COUNT="$(printf "%s" "$GPU_COUNT_RAW" | tr -cd "0-9")"
if [[ -z "$GPU_COUNT" ]]; then
  GPU_COUNT=1
fi

THREADS_PER_WORKER=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$SLURM_CPUS_PER_TASK" -gt 0 ]]; then
  THREADS_PER_WORKER=$((SLURM_CPUS_PER_TASK / GPU_COUNT))
  if [[ "$THREADS_PER_WORKER" -lt 1 ]]; then
    THREADS_PER_WORKER=1
  fi
fi

export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "$MODEL" ]]; then
  MODEL="$(
    .venv/bin/python -u - <<PY
from config.settings import CONFIG
print(CONFIG.benchmark("level${LEVEL}.model", "") or "")
PY
  )"
fi

if [[ -z "$MODEL" ]]; then
  echo "MODEL is empty. Set MODEL via sbatch --export or config/config.yaml benchmark.level${LEVEL}.model." >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job Name: ${SLURM_JOB_NAME:-socialomni}"
echo "Node: $(hostname)"
echo "Submit Dir: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "GPU_COUNT: $GPU_COUNT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MODEL: $MODEL"
echo "LEVEL: $LEVEL"
echo "MODALITY: ${MODALITY:-${SOCIALOMNI_MODALITY:-config_default}}"
echo "MAX_SAMPLES: ${MAX_SAMPLES:-all}"
echo "REUSE_SERVER: $REUSE_SERVER"
date

nvidia-smi || true

CMD=(
  .venv/bin/python -u scripts/run_test_with_autoserver.py
  --model "$MODEL"
  --level "$LEVEL"
  --server-timeout "$SERVER_TIMEOUT"
  --test-timeout "$TEST_TIMEOUT"
  --progress-interval "$PROGRESS_INTERVAL"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  CMD+=(--resume)
fi
if [[ "$REUSE_SERVER" == "1" || "$REUSE_SERVER" == "true" ]]; then
  CMD+=(--reuse-server)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

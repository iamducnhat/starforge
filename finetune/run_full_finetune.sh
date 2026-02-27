#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 finetune/generate_synthetic_tool_use.py --output finetune/synthetic_tool_use.jsonl --per-topic 120
python3 finetune/build_function_dataset.py --output finetune/train_function_tools.jsonl
python3 finetune/build_full_tool_dataset.py \
  --train-output finetune/train_tool_use_full.jsonl \
  --val-output finetune/val_tool_use_full.jsonl \
  --val-ratio 0.02

python3 finetune/train_lora_sft.py \
  --model "${FINETUNE_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --train-file finetune/train_tool_use_full.jsonl \
  --val-file finetune/val_tool_use_full.jsonl \
  --output-dir "${FINETUNE_OUTPUT_DIR:-finetune/output/lora_tool_use}" \
  --epochs "${FINETUNE_EPOCHS:-2}" \
  --batch-size "${FINETUNE_BATCH_SIZE:-1}" \
  --grad-accum "${FINETUNE_GRAD_ACCUM:-16}" \
  --max-length "${FINETUNE_MAX_LENGTH:-2048}" \
  --lr "${FINETUNE_LR:-2e-4}" \
  --bf16

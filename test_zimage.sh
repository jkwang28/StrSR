#!/usr/bin/env bash

set -euo pipefail

exec python test_zimage.py \
  --input_lq testdata/LR \
  --output results/zimage \
  --base_model_path pretrained/Z-Image-Turbo \
  --qwen_model_path pretrained/Qwen3-VL-4B-Instruct \
  --weight_path pretrained/StrSR-zimage \
  --tile 1024 \
  --overlap 64 \
  --precision bf16 \
  --model_t 800 \
  --conditioning qwen \
  --coeff_t 800 \
  --lora_rank 256 \
  --lora_modules to_q to_k to_v to_out.0 feed_forward.w1 feed_forward.w2 feed_forward.w3 \
  "$@"

#!/usr/bin/env bash

set -euo pipefail

exec python test_flux.py \
  --input_lq testdata/LR \
  --output results/flux \
  --base_model_path pretrained/FLUX.2-klein-base-4B \
  --qwen_model_path pretrained/Qwen3-VL-4B-Instruct \
  --weight_path pretrained/StrSR-flux \
  --tile 1024 \
  --overlap 64 \
  --precision bf16 \
  --model_t 800 \
  --conditioning qwen \
  --coeff_t 800 \
  --lora_rank 256 \
  --lora_modules to_q to_k to_v add_k_proj add_q_proj add_v_proj to_add_out to_out to_out.0 to_qkv_mlp_proj \
  "$@"

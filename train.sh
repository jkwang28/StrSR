#!/usr/bin/bash

set -euo pipefail

TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --mixed_precision=bf16 train.py \
    --config configs/flux_train_stage1.yaml

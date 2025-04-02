#!/usr/bin/zsh

# model_type must be in ["lraspp_mobilenet_v3_large", "fast_scnn", "bisenetv2", "u2net_full", "u2net_lite"]

model_type="bisenetv2"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
                             --nproc_per_node=2 \
                             train.py \
                             --model_type $model_type

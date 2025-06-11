#!/usr/bin/bash

# model_type must be in ["lraspp_mobilenet_v3_large", ""lraspp_mobilenet_v3_small", "fast_scnn", "bisenetv2", "u2net_full", "u2net_lite"]

model_type="lraspp_mobilenet_v3_small"
num_epochs=400
ckpt_path="/algdata03/wei.wei/data/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250411_124619/lraspp_mobilenet_v3_large_254_iou_0.9229.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                             --nproc_per_node=4 \
                             train.py \
                             --model_type $model_type \
                             --num_epochs $num_epochs \
                            #  --ckpt_path $ckpt_path 

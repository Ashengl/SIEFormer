#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name 'scars' \
    --batch_size 128 \
    --transform 'imagenet' \
    --eval_funcs 'v2' \
    --filter 1

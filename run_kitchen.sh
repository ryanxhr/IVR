#!/bin/bash

# Script to reproduce results

GPU_LIST=(1 0)

for seed in 0; do
for alg in "SQL"; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "kitchen-complete-v0" \
  --config=configs/kitchen_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "kitchen-partial-v0" \
  --config=configs/kitchen_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "kitchen-mixed-v0" \
  --config=configs/kitchen_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"


done
done
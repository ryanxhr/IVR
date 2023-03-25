#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

env_list=(
	"kitchen-complete-v0"
	"kitchen-partial-v0"
	"kitchen-mixed-v0"
	)

for seed in 77; do
for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/kitchen_config.py \
  --alg "SQL" \
  --alpha 0.5 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/kitchen_config.py \
  --alg "EQL" \
  --alpha 2.0 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
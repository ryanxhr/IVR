#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

env_list=(
	"hopper-medium-v2"
	"halfcheetah-medium-v2"
	"walker2d-medium-v2"
	"hopper-medium-replay-v2"
	"halfcheetah-medium-replay-v2"
	"walker2d-medium-replay-v2"
	"hopper-medium-expert-v2"
	"halfcheetah-medium-expert-v2"
	"walker2d-medium-expert-v2"
	)

for seed in 77; do
for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/mujoco_config.py \
  --alg "SQL" \
  --alpha 1.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/mujoco_config.py \
  --alg "EQL" \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
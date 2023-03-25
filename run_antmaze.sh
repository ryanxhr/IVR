#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

env_list=(
	"antmaze-umaze-v2"
#	"antmaze-umaze-diverse-v2"
	"antmaze-medium-play-v2"
	"antmaze-medium-diverse-v2"
	"antmaze-large-play-v2"
	"antmaze-large-diverse-v2"
	)

for seed in 77; do

for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/antmaze_config.py \
  --alg "SQL" \
  --alpha 0.1 \
  --eval_interval 100000 \
  --eval_episodes 100 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name $env \
  --config=configs/antmaze_config.py \
  --alg "EQL" \
  --alpha 0.5 \
  --eval_interval 100000 \
  --eval_episodes 100 \
  --seed $seed &

sleep 2
let "task=$task+1"

done

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "antmaze-umaze-diverse-v2" \
  --config=configs/antmaze_config.py \
  --alg "SQL" \
  --alpha 3.0 \
  --eval_interval 100000 \
  --eval_episodes 100 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "antmaze-umaze-diverse-v2" \
  --config=configs/antmaze_config.py \
  --alg "EQL" \
  --alpha 5.0 \
  --eval_interval 100000 \
  --eval_episodes 100 \
  --seed $seed &

sleep 2
let "task=$task+1"


done
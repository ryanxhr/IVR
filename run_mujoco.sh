#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3 4 5 6 7)

for seed in 0 1 2 3 4; do
for alg in "SQL" "EQL"; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "halfcheetah-medium-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "hopper-medium-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "walker2d-medium-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "halfcheetah-medium-replay-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "hopper-medium-replay-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "walker2d-medium-replay-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 2.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "halfcheetah-medium-expert-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 5.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "hopper-medium-expert-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 5.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
  --env_name "walker2d-medium-expert-v2" \
  --config=configs/mujoco_config.py \
  --alg $alg \
  --alpha 5.0 \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
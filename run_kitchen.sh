#!/bin/bash

# Script to reproduce results
GPU_LIST=(1)

#for ((i=0;i<10;i+=1))
for i in 0; do
#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-umaze-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#	--eval_interval 10000 \
#	--seed $i \
#	--tmp 0.5 &
#
#  sleep 2
#  let "task=$task+1"
#
#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-umaze-diverse-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#  --eval_interval 10000 \
#	--seed $i \
#	--tmp 5.0 &
#
#  sleep 2
#  let "task=$task+1"
#
#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-medium-play-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#	--eval_interval 10000 \
#	--seed $i \
#	--tmp 0.5 &
#
#  sleep 2
#  let "task=$task+1"
#
#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-medium-diverse-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#	--eval_interval 10000 \
#	--seed $i \
#	--tmp 0.5 &
#
#  sleep 2
#  let "task=$task+1"

#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-large-play-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#	--eval_interval 10000 \
#	--seed $i \
#	--tmp 0.5 &
#
#  sleep 2
#  let "task=$task+1"
#
#  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
#	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
#	--config configs/antmaze_config.py \
#	--env_name "antmaze-large-diverse-v2"  \
#	--alg "EQL" \
#	--eval_episodes 100 \
#	--eval_interval 10000 \
#	--seed $i \
#	--tmp 0.5 &
#
#  sleep 2
#  let "task=$task+1"

  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
	--config configs/kitchen_config.py \
	--env_name "kitchen-complete-v0"  \
	--alg "EQL" \
	--eval_episodes 10 \
	--eval_interval 10000 \
	--seed $i \
	--tmp 2.0 &

  sleep 2
  let "task=$task+1"

  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
	--config configs/kitchen_config.py \
	--env_name "kitchen-partial-v0"  \
	--alg "EQL" \
	--eval_episodes 10 \
	--eval_interval 10000 \
	--seed $i \
	--tmp 2.0 &

  sleep 2
  let "task=$task+1"

  GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
	CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
	--config configs/kitchen_config.py \
	--env_name "kitchen-mixed-v0"  \
	--alg "EQL" \
	--eval_episodes 10 \
	--eval_interval 10000 \
	--seed $i \
	--tmp 2.0 &

  sleep 2
  let "task=$task+1"

done
#!/bin/bash
declare -a envs=("antmaze-large-diverse-v2" "antmaze-large-play-v2")
# "antmaze-umaze-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-diverse-v2" "antmaze-medium-play-v2" "antmaze-large-diverse-v2" "antmaze-large-play-v2"
# Script to reproduce results
CUDA_VISIBLE_DEVICES=1
seeds=3
# for seed in 1 2 3; do
for env in "${envs[@]}"; do
for seed in $(seq 0 $((seeds-1))); do
python train_offline.py \
    --env_name=$env \
    --config=configs/antmaze_config.py \
    --eval_episodes=100 \
    --eval_interval=100000 \
    --tmp=0.05 \
    --seed=$seed
done
done
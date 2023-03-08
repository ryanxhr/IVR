#!/bin/bash

pids=()
# {0..5}
for i in {42..43}
do
    CUDA_VISIBLE_DEVICES=1 python train_offline.py \
    --config=configs/antmaze_config.py \
    --env_name=antmaze-large-diverse-v2 \
    --eval_episodes=100 \
    --eval_interval=100000 \
    --tmp=0.5 \
    --seed=$i &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done


#!/bin/bash

# pids=()
# {0..5}
for i in {5..7}; do
    python train_offline.py \
    --config=configs/mujoco_config.py \
    --env_name=halfcheetah-medium-v2 \
    --tmp=1 \
    --seed=$i
    # pids+=( "$!" )
    # sleep 5 # add 5 second delay
done

# for pid in "${pids[@]}"; do
#     wait $pid
# done
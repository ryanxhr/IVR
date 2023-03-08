#!/bin/bash

# pids=()
# {0..5}
for i in {0..4}; do
    python train_offline.py \
    --config=configs/mujoco_config.py \
    --env_name=hopper-medium-replay-v2 \
    --tmp=2 \
    --seed=$i
    # pids+=( "$!" )
    # sleep 5 # add 5 second delay
done

# for pid in "${pids[@]}"; do
#     wait $pid
# done
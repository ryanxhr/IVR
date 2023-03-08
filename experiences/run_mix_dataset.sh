#!/bin/bash

# pids=()
# {0..5}
for i in 0.01 0.5 0.1 0.2 0.3; do
    python train_offline_mix_sql.py \
    --env_name=walker2d-expert-v2 \
    --config=configs/mujoco_config.py \
    --tmp=10 \
    --mix_dataset=random \
    --expert_ratio=$i
done



# for pid in "${pids[@]}"; do
#     wait $pid
# done
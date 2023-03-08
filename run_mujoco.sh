#!/bin/bash

# Script to reproduce results


for tmp in 0.5 1 2 5 10; do
python train_offline.py \
    --env_name=halfcheetah-medium-expert-v2 \
    --config=configs/mujoco_config.py \
    --tmp=$tmp \
    --seed=1
done
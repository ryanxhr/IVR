# Offline Reinforcement Learning with Implicit Q-Learning

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

For a PyTorch reimplementation see https://github.com/rail-berkeley/rlkit/tree/master/examples/iql

## How to run the code

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion
```bash
CUDA_VISIBLE_DEVICES=1 python train_offline.py --env_name=hopper-medium-replay-v2 --config=configs/mujoco_config.py --tmp=2
```

AntMaze
```bash
CUDA_VISIBLE_DEVICES=1 python train_offline.py --env_name=antmaze-medium-diverse-v2 --config=configs/antmaze_config.py --eval_episodes=50 --tmp=0.5 --alg=EQL
```

Kitchen and Adroit
```bash
CUDA_VISIBLE_DEVICES=1 python train_offline.py --env_name=kitchen-partial-v0 --config=configs/kitchen_config.py --tmp=5
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

## Misc
## Mix dataset
Please Note the learner's version (from IQL to SQL)
```bash
CUDA_VISIBLE_DEVICES=1 nohup python train_offline_mix_sql.py --env_name=walker2d-expert-v2 --config=configs/mujoco_config.py --tmp=10 --mix_dataset=random --expert_ratio=0.1
```

## Few samples dataset
```bash
CUDA_VISIBLE_DEVICES=1 nohup python train_offline_few_sample_sql.py --env_name=antmaze-medium-play-v2 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --tmp=0.5 --heavy_tail_higher=0.1 --seed 42
```

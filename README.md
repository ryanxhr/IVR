# Sparse Q-Learning: Offline Reinforcement Learning with Implicit Value Regularization

This is the code for reproducing the results of the paper Sparse Q-Learning: Offline Reinforcement Learning with Implicit Value Regularization accepted as **oral** at ICLR'2023. The paper and slide can be found at [paper](https://arxiv.org/abs/2210.08323) and [slide](https://docs.google.com/presentation/d/1swZTLDSvZLGCrXs46tzSHLWZC6VfO9qYChegjjadCpc/edit#slide=id.g170ea50d4c3_9_42).


### Usage
Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The paper results can be reproduced by running:
```
./run_sql.sh
```


### Bibtex
```
@inproceedings{xu2022sparse,
  title  = {Sparse Q-Learning: Offline Reinforcement Learning with Implicit Value Regularization},
  author = {Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, Xianyuan Zhan},
  year   = {2022},
  booktitle = {Advances in Neural Information Processing Systems},
}
```
# Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization

This is the code for reproducing the results of the paper Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization accepted as **Notable-top-5%** at ICLR'2023. The paper and slide can be found at [paper](https://arxiv.org/abs/2210.08323) and [slide](https://docs.google.com/presentation/d/1swZTLDSvZLGCrXs46tzSHLWZC6VfO9qYChegjjadCpc/edit#slide=id.g170ea50d4c3_9_42).


### Usage
Our code is built on the jax version code of IQL (https://github.com/ikostrikov/implicit_q_learning). The paper results can be reproduced by running:
```
./run.sh
```


### Bibtex
```
@inproceedings{xu2023offline,
  title  = {Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization},
  author = {Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, Xianyuan Zhan},
  year   = {2023},
  booktitle = {International Conference on Learning Representations},
}
```
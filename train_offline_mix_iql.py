import os
from typing import Tuple
from pathlib import Path
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from dataset_utils import Log
import wandb

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from iql_learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'walker2d-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './results/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float('tmp', 1, 'hyper')
flags.DEFINE_string('mix_dataset', 'None', 'mix the dataset')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_float('expert_ratio', 0.1 , 'the expert dataset ratio')
config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int,
                         mix_dataset: str=None,
                         expert_ratio: int=1. ) -> Tuple[gym.Env, D4RLDataset]:
    if 'expert' not in env_name:
        raise ValueError('the env_name must be expert')

    base_env_dict = ['hopper', 'walker2d', 'halfcheetah']
    add_env_name = [base_env for base_env in base_env_dict if base_env in env_name]
    add_env_name = add_env_name[0]
    if mix_dataset=='medium':
        add_env_name = f"{add_env_name}-medium-v2"
    elif mix_dataset=='random':
        add_env_name = f"{add_env_name}-random-v2"
    else:
        raise NameError('mix_dataset must be random or medium')
    "mix the medium with expert"
    add_env = gym.make(add_env_name)
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    dataset = D4RLDataset(env, add_env=add_env, expert_ratio=expert_ratio)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.mix_dataset, FLAGS.expert_ratio)
    kwargs = dict(FLAGS.config)
    kwargs['tmp']=FLAGS.tmp
    kwargs['mix_dataset']=FLAGS.mix_dataset
    log = Log(Path('mix_data_IQL')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    # log(f'Total target location reward {dataset.rewards.sum() + len(dataset.rewards)}')
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)
    kwargs['expert_ratio']=FLAGS.expert_ratio
    wandb.init(
        project='SQL_mix_data',
        entity='louis_t0',
        name=f"{FLAGS.env_name}_iql",
        config=kwargs
    )

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            wandb.log(update_info, i)

        if i % FLAGS.eval_interval == 0:
            normalized_return = evaluate(FLAGS.env_name, agent, env, FLAGS.eval_episodes)
            log.row({'normalized_return': normalized_return})
            wandb.log({'normalized_return': normalized_return}, i)


if __name__ == '__main__':
    app.run(main)

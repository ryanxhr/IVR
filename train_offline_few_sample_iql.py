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

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './results/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float('tmp', 1, 'hyper')
flags.DEFINE_string('mix_dataset', 'None', 'mix the dataset')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

flags.DEFINE_boolean('heavy_tail', True, 'the few samples settings')
flags.DEFINE_float('heavy_tail_higher', 0., 'the few samples settings')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
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
                         heavy_tail: bool,
                         heavy_tail_higher: float) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    dataset = D4RLDataset(env, heavy_tail=heavy_tail, heavy_tail_higher=heavy_tail_higher)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        # pass
        normalize(dataset)

    return env, dataset


def main(_):
    env, dataset = make_env_and_dataset(FLAGS.env_name,
                                    FLAGS.seed,
                                    heavy_tail=FLAGS.heavy_tail,
                                    heavy_tail_higher=FLAGS.heavy_tail_higher)
    kwargs = dict(FLAGS.config)
    kwargs['tmp']=FLAGS.tmp
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    kwargs['heavy_tail_higher']=FLAGS.heavy_tail_higher
    log = Log(Path('few_sample_IQL')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    log(f'Total target location reward {dataset.rewards.sum() + len(dataset.rewards)}')
    wandb.init(
        project='SQL_few_samples_',
        entity='louis_t0',
        name=f"IQL_{FLAGS.env_name}",
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

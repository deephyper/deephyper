import datetime
import glob
import os
import pickle
import signal
import sys
import argparse
import json
from collections import OrderedDict
from math import ceil, log
from pprint import pprint, pformat
from random import random
from time import ctime, time, sleep
from importlib import import_module, reload
import numpy as np
import tensorflow as tf

from deephyper.evaluators import Evaluator
from deephyper.search import util

from tensorforce.agents import PPOAgent
from tensorforce.agents import RandomAgent
from tensorforce.execution import DistributedRunner
from tensorforce.environments import AsyncNasBalsamEnvironment

logger = util.conf_logger('deephyper.search.run_nas')


def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))

def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))

class Search:

    def __init__(self, **kwargs):
        self.run_func = util.load_attr_from(kwargs.get('run'))
        self.num_episodes = kwargs.get('num_episodes')
        self.problem = util.load_attr_from(f'{kwargs.get("problem")}.problem.Problem')()
        self.space = self.problem.space
        self.evaluator = Evaluator.create(self.run_func, cache_key=key, method=args.evaluator)
        logger.debug(f'evaluator: {type(self.evaluator)}')
        self.structure = None
        self.num_workers = kwargs.get('nodes')
        self.batch_size = args.batch_size
        self.update_freq = args.update_freq
        self.agent_type = args.agent
        logger.debug(f'update mode: (batch_size={self.batch_size}, '
                     f'frequency={self.update_freq})')

    def run(self):
        # Settings
        num_parallel = self.evaluator.num_workers - 4 #balsam launcher & controller of search
        num_episodes = self.num_episodes
        logger.debug(f'num_parallel: {num_parallel}')
        logger.debug(f'num_episodes: {num_episodes}')

        # stub structure to know how many nodes we need to compute
        logger.debug('create structure')
        self.structure = self.space['create_structure']['func'](
            tf.constant([[1., 1.]]),
            **self.space['create_structure']['kwargs']
        )

        # Creating the environment
        logger.debug('create environment')
        environment = AsyncNasBalsamEnvironment(self.space, self.evaluator, self.structure)

        # Creating the Agent
        network_spec = [
            dict(type='internal_lstm', size=32)
        ]

        logger.debug('create agent')
        if self.agent_type = 'ppo':
            agent = PPOAgent(
                states=environment.states,
                actions=environment.actions,
                network=network_spec,
                # Agent
                states_preprocessing=None,
                actions_exploration=None,
                reward_preprocessing=None,
                # MemoryModel
                update_mode=dict(
                    unit='episodes',
                    # 10 episodes per update
                    batch_size=self.batch_size,
                    # Every 10 episodes
                    frequency=self.update_freq,
                ),
                memory=dict(
                    type='latest',
                    include_next_states=False,
                    capacity=5000
                ),
                # DistributionModel
                distributions=None,
                entropy_regularization=0.01,
                # PGModel
                baseline_mode='states',
                baseline=dict(
                    type='mlp',
                    sizes=[32, 32]
                ),
                baseline_optimizer=dict(
                    type='multi_step',
                    optimizer=dict(
                        type='adam',
                        learning_rate=1e-3
                    ),
                    num_steps=5
                ),
                gae_lambda=0.97,
                # PGLRModel
                likelihood_ratio_clipping=0.2,
                # PPOAgent
                step_optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                subsampling_fraction=0.2,
                optimization_steps=25,
                execution=dict(
                    type='single',
                    num_parallel=num_parallel,
                    session_config=None,
                    distributed_spec=None
                )
            )
        elif self.agent_type == 'rdm':
            agent = RandomAgent(
                states=environment.states,
                actions=environment.actions
            )
        else:
            raise Exception(f'Invalid agent type :{self.agent_type}')

        # Creating the Runner
        runner = DistributedRunner(agent=agent, environment=environment)
        runner.run(num_episodes=num_episodes, episode_finished=print_logs)
        runner.close()


def main(args):
    '''Service loop: add jobs; read results; drive nas'''
    kwargs = vars(args)
    logger.debug(f'args: {pformat(kwargs)}')
    controller = Search(**kwargs)
    controller.run()

def create_parser():
    """Command line parser for NAS"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--evaluator',
                        default='local',
                        help="must be 'local' or 'balsam'")
    parser.add_argument("--problem",
                        default="deephyper.benchmarks.linearRegNas",
                        help="")
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='maximum number of episodes')
    parser.add_argument('--nodes', type=int, default=None)
    parser.add_argument('--run',
                        default="deephyper.run.nas_structure_raw.run",
                        help='ex. deephyper.run.nas_structure_raw.run')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--update_freq', type=int, default=10)
    parser.add_argument('--agent', default='ppo')

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

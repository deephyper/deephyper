import datetime
import glob
import os
import pickle
import signal
import sys
from collections import OrderedDict
from math import ceil, log
from pprint import pprint
from random import random
from time import ctime, time, sleep
from importlib import import_module, reload
import numpy as np
import tensorflow as tf
from multiprocessing.pool import ThreadPool

from deephyper.evaluators import evaluate
from deephyper.search import util

from tensorforce.agents import PPOAgent
from tensorforce.execution import DistributedRunner
from tensorforce.environments import AsyncNasBalsamEnvironment

logger = util.conf_logger('deephyper.search.run_nas')

# import subprocess as sp
# logger.debug(f'ddd {sp.Popen("which mpirun".split())}')
# logger.debug(f'python exe : {sys.executable}')

def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))

class Search:

    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator_nas(cfg)
        self.config = cfg.config
        self.map_model_reward = {}

    def run(self):
        # Settings
        num_parallel = self.opt_config.num_workers
        num_episodes = None

        # Creating the environment
        environment = AsyncNasBalsamEnvironment(self.opt_config)

        # Creating the Agent
        network_spec = [
            dict(type='internal_lstm', size=32)
        ]

        agent = PPOAgent(
            states=environment.states,
            actions=environment.actions,
            network=network_spec,
            # Agent
            states_preprocessing=None,
            actions_exploration=None,
            reward_preprocessing=None,
            # batching_capacity=10,
            # MemoryModel
            update_mode=dict(
                unit='episodes',
                # 10 episodes per update
                batch_size=1,
                # Every 10 episodes
                frequency=1,
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

        # Creating the Runner
        runner = DistributedRunner(agent=agent, environment=environment)
        runner.run(num_episodes=num_episodes, episode_finished=print_logs)
        runner.close()


def main(args):
    '''Service loop: add jobs; read results; drive nas'''
    cfg = util.OptConfigNas(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS on benchmark {cfg.benchmark} & run with {cfg.run_module_name}")
    controller.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

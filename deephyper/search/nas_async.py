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
from importlib import import_module
from mpi4py import MPI
import numpy as np
import tensorflow as tf

from deephyper.evaluators import Evaluator
from deephyper.search import util

from gym_nas.agent.run_nas_async import train

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
        self.num_agents = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.lr = args.lr

    def run(self):
        # Settings
        #num_parallel = self.evaluator.num_workers - 4 #balsam launcher & controller of search for cooley
        num_nodes = self.evaluator.num_workers - 1 #balsam launcher & controller of search for theta
        if num_nodes > self.num_agents:
            num_episodes_per_batch = (num_nodes-self.num_agents)//self.num_agents
        else:
            num_episodes_per_batch = 1

        if self.rank == 0:
            logger.debug(f'<Rank={self.rank}> num_nodes: {num_nodes}')
            logger.debug(f'<Rank={self.rank}> num_episodes_per_batch: {num_episodes_per_batch}')

        logger.debug(f'<Rank={self.rank}> starting training...')
        train(
            num_episodes=self.num_episodes,
            seed=2018,
            space=self.problem.space,
            evaluator=self.evaluator,
            num_episodes_per_batch=num_episodes_per_batch
        )


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
    parser.add_argument('--run',
                        default="deephyper.run.nas_structure_raw.run",
                        help='ex. deephyper.run.nas_structure_raw.run')
    parser.add_argument('--lr', type=float, default=1e-3)

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

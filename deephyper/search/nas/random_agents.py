import json
import math
import os
from math import ceil, log
from pprint import pformat, pprint

import tensorflow as tf
from mpi4py import MPI

from deephyper.search import Search, util
from deephyper.search.nas.agent import nas_random

logger = util.conf_logger('deephyper.search.run_nas')


def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))


def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))


LAUNCHER_NODES = int(os.environ.get('BALSAM_LAUNCHER_NODES', 1))
WORKERS_PER_NODE = int(os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))


class RandomAgents(Search):
    """Neural Architecture search using random search.
    """

    def __init__(self, problem, run, evaluator, **kwargs):
        self.rank = MPI.COMM_WORLD.Get_rank()
        if self.rank == 0:
            super().__init__(problem, run, evaluator, cache_key=key, **kwargs)
        MPI.COMM_WORLD.Barrier()
        if self.rank != 0:
            super().__init__(problem, run, evaluator, cache_key=key, **kwargs)
        # set in super : self.problem
        # set in super : self.run_func
        # set in super : self.evaluator

        self.num_episodes = kwargs.get('num_episodes')
        if self.num_episodes is None:
            self.num_episodes = math.inf
        self.space = self.problem.space
        logger.debug(f'evaluator: {type(self.evaluator)}')
        self.num_agents = MPI.COMM_WORLD.Get_size() - 1  # one is  the parameter server
        logger.debug(f'num_agents: {self.num_agents}')
        logger.debug(f'rank: {self.rank}')

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--num-episodes', type=int, default=None,
                            help='maximum number of episodes')
        return parser

    def main(self):
        num_nodes = LAUNCHER_NODES * WORKERS_PER_NODE  # balsam launcher
        if num_nodes > self.num_agents:
            num_episodes_per_batch = (
                num_nodes-self.num_agents)//self.num_agents
        else:
            num_episodes_per_batch = 1

        if self.rank == 0:
            logger.debug(f'<Rank={self.rank}> num_nodes: {num_nodes}')
            logger.debug(
                f'<Rank={self.rank}> num_episodes_per_batch: {num_episodes_per_batch}')

        logger.debug(f'<Rank={self.rank}> starting training...')

        nas_random.train(
            num_episodes=self.num_episodes,
            seed=2018,
            space=self.problem.space,
            evaluator=self.evaluator,
            num_episodes_per_batch=num_episodes_per_batch
        )


if __name__ == "__main__":
    args = NasRandom.parse_args()
    search = NasRandom(**vars(args))
    search.main()

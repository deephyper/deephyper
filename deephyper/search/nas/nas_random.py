import os
import json
from math import ceil, log
from pprint import pprint, pformat
from mpi4py import MPI
import math
import tensorflow as tf

from deephyper.evaluators import Evaluator
from deephyper.search import util, Search

from tensorforce.agents import RandomAgent
from tensorforce.execution import DistributedRunner
from tensorforce.environments import AsyncNasBalsamEnvironment

logger = util.conf_logger('deephyper.search.run_nas')

def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))

def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))

LAUNCHER_NODES = int(os.environ.get('BALSAM_LAUNCHER_NODES', 1))
WORKERS_PER_NODE = int(os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))

class NasRandom(Search):
    def __init__(self, problem, run, evaluator, **kwargs):
        super().__init__(problem, run, evaluator, **kwargs)
        # set in super : self.problem
        # set in super : self.run_func
        # set in super : self.evaluator
        self.evaluator = Evaluator.create(self.run_func,
                                          cache_key=key,
                                          method=evaluator)
        self.num_episodes = kwargs.get('num_episodes')
        if self.num_episodes is None:
            self.num_episodes = math.inf
        self.space = self.problem.space
        logger.debug(f'evaluator: {type(self.evaluator)}')

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--num-episodes', type=int, default=None,
                            help='maximum number of episodes')
        return parser

    def main(self):
        # Settings
        #num_parallel = self.evaluator.num_workers - 4 #balsam launcher & controller of search for cooley
        # num_nodes = self.evaluator.num_workers - 1 #balsam launcher & controller of search for theta
        num_parallel = LAUNCHER_NODES * WORKERS_PER_NODE - 2 # balsam launcher and controller
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
        environment = AsyncNasBalsamEnvironment(self.space, self.evaluator, self.structure, mode='full')

        agent = RandomAgent(
            states=environment.states,
            actions=environment.actions,
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

if __name__ == "__main__":
    args = NasRandom.parse_args()
    search = NasRandom(**vars(args))
    search.main()

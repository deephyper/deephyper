import json
from math import ceil, log
from pprint import pprint, pformat
from mpi4py import MPI
import math

from deephyper.evaluators import Evaluator
from deephyper.search import util, Search

# from gym_nas.agent.run_nas_async import train
from gym_nas.agent.nas_ppo_async_a3c import train

logger = util.conf_logger('deephyper.search.run_nas')

def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))

def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))

class NasPPOAsyncA3C(Search):
    def __init__(self, problem, run, evaluator, **kwargs):
        super().__init__(problem, run, evaluator, **kwargs)
        # set in super : self.problem
        # set in super : self.run_func
        # set in super : self.evaluator
        self.evaluator = Evaluator.create(self.run_func,
                                          cache_key=key,
                                          method=kwargs['evaluator'])
        self.num_episodes = kwargs.get('num_episodes')
        if self.num_episodes is None:
            self.num_episodes = math.inf
        self.space = self.problem.space
        logger.debug(f'evaluator: {type(self.evaluator)}')
        self.num_agents = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.lr = args.lr

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--num-episodes', type=int, default=None,
                            help='maximum number of episodes')
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def main(self):
        # Settings
        #num_parallel = self.evaluator.num_workers - 4 #balsam launcher & controller of search for cooley
        num_nodes = self.evaluator.num_workers - 1 #balsam launcher & controller of search for theta
        num_nodes -= 1 # parameter server is neither an agent nor a worker
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

if __name__ == "__main__":
    args = NasPPOAsyncA3C.parse_args()
    search = NasPPOAsyncA3C(**vars(args))
    search.main()

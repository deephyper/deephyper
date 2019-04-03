import os
import json
from math import ceil, log
from pprint import pprint, pformat
from mpi4py import MPI
import math

from deephyper.evaluator import Evaluator
from deephyper.search import util, Search

from deephyper.search.nas.agent import nas_ppo_async_a3c

logger = util.conf_logger('deephyper.search.nas.ppo_a3c_async')

def print_logs(runner):
    logger.debug('num_episodes = {}'.format(runner.global_episode))
    logger.debug(' workers = {}'.format(runner.workers))

def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))

LAUNCHER_NODES = int(os.environ.get('BALSAM_LAUNCHER_NODES', 1))
WORKERS_PER_NODE = int(os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))

class NasPPOAsyncA3C(Search):
    """Neural Architecture search using proximal policy gradient with asynchronous optimization.
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
        # self.evaluator = Evaluator.create(self.run_func,
        #                                   cache_key=key,
        #                                   method=evaluator)

        self.num_episodes = kwargs.get('num_episodes')
        if self.num_episodes is None:
            self.num_episodes = math.inf
        self.clip_param = kwargs.get('clip_param')
        self.entcoeff = kwargs.get('entcoeff')
        self.optim_epochs = kwargs.get('optim_epochs')
        self.optim_stepsize = kwargs.get('optim_stepsize')
        self.optim_batch_size = kwargs.get('optim_batchsize')
        self.gamma = kwargs.get('gamma')
        self.lam = kwargs.get('lam')

        self.reward_rule = util.load_attr_from('deephyper.search.nas.agent.utils.'+kwargs['reward_rule'])

        self.space = self.problem.space

        logger.debug(f'evaluator: {type(self.evaluator)}')

        self.num_agents = MPI.COMM_WORLD.Get_size() - 1 # one is  the parameter server

        logger.debug(f'num_agents: {self.num_agents}')
        logger.debug(f'rank: {self.rank}')

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--num-episodes', type=int, default=None,
                            help='maximum number of episodes')
        parser.add_argument('--reward-rule', type=str,
            default='reward_for_final_timestep',
            choices=[
                'reward_for_all_timesteps',
                'reward_for_final_timestep'
            ],
            help='A function which describe how to spread the episodic reward on all timesteps of the corresponding episode.')
        parser.add_argument('--clip-param', type=float, default=0.2)
        parser.add_argument('--entcoeff', type=float, default=0.01)
        parser.add_argument('--optim-epochs', type=int, default=4)
        parser.add_argument('--optim-stepsize', type=float, default=1e-3)
        parser.add_argument('--optim-batchsize', type=int, default=None)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--lam', type=float, default=0.95)
        return parser

    def main(self):
        # Settings
        num_nodes = LAUNCHER_NODES * WORKERS_PER_NODE - 1 # for PS
        if num_nodes > self.num_agents:
            num_episodes_per_batch = (num_nodes-self.num_agents)//self.num_agents
        else:
            num_episodes_per_batch = 1

        if self.rank == 0:
            logger.debug(f'<Rank={self.rank}> num_nodes: {num_nodes}')
            logger.debug(f'<Rank={self.rank}> num_episodes_per_batch: {num_episodes_per_batch}')

        logger.debug(f'<Rank={self.rank}> starting training...')
        nas_ppo_async_a3c.train(
            num_episodes=self.num_episodes,
            seed=2018,
            space=self.problem.space,
            evaluator=self.evaluator,
            num_episodes_per_batch=num_episodes_per_batch,
            reward_rule=self.reward_rule,
            clip_param=self.clip_param,
            entcoeff=self.entcoeff,
            optim_epochs=self.optim_epochs,
            optim_stepsize=self.optim_stepsize,
            optim_batchsize=self.optim_batch_size,
            gamma=self.gamma,
            lam=self.lam
        )

if __name__ == "__main__":
    args = NasPPOAsyncA3C.parse_args()
    search = NasPPOAsyncA3C(**vars(args))
    search.main()

import json
import math
import sys
import time
import datetime
from importlib import import_module


from deephyper.search import Search, util
from deephyper.search.nas.baselines import logger
from deephyper.search.nas.baselines.common.cmd_util import (common_arg_parser,
                                                            make_env,
                                                            make_vec_env,
                                                            parse_unknown_args)
from deephyper.search.nas.baselines.common.tf_util import get_session
from deephyper.search.nas.baselines.common.vec_env import (VecEnv,
                                                           VecFrameStack,
                                                           VecNormalize)
from deephyper.search.nas.baselines.common.vec_env.vec_video_recorder import \
    VecVideoRecorder
from deephyper.search.nas.env.neural_architecture_envs import \
    NeuralArchitectureVecEnv
from deephyper.search.nas.utils._logging import JsonMessage as jm

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


dhlogger = util.conf_logger('deephyper.search.nas.nas_search')


def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))


class NeuralArchitectureSearch(Search):

    def __init__(self, problem, run, evaluator,  alg, network, num_envs,
                 **kwargs):
        """Represents different kind of RL algorithms working with NAS.

        Args:
            problem (Problem): problem instance to use.
            run (function): function to use for evaluations.
            evaluator ([Evaluator): evaluator to use.
            alg (str): algorithm to use among ['ppo2',].
            network (str/function): policy network.
            num_envs (int): number of environments per agent to run in
                parallel, it corresponds to the number of evaluation per
                batch per agent.
        """

        self.kwargs = kwargs
        if MPI is None:
            dhlogger.info(jm(
                type='start_infos',
                alg=alg,
                network=network,
                num_envs_per_agent=num_envs
                nagents=1))
            self.rank = 0
            super().__init__(problem, run, evaluator, cache_key=key, **kwargs)
        else:
            self.rank = MPI.COMM_WORLD.Get_rank()
            if self.rank == 0:
                dhlogger.info(jm(
                    type='start_infos',
                    alg=alg,
                    network=network,
                    num_envs_per_agent=num_envs
                    nagents=MPI.COMM_WORLD.Get_size()))
                super().__init__(problem, run, evaluator, cache_key=key,
                                 **kwargs)
            MPI.COMM_WORLD.Barrier()
            if self.rank != 0:
                super().__init__(problem, run, evaluator, cache_key=key,
                                 **kwargs)
        # set in super : self.problem, self.run_func, self.evaluator

        self.num_evals = kwargs.get('max_evals')
        if self.num_evals is None:
            self.num_evals = math.inf

        self.space = self.problem.space

        dhlogger.info(f'evaluator: {type(self.evaluator)}')
        dhlogger.info(f'rank: {self.rank}')
        dhlogger.info(f'alg: {alg}')
        dhlogger.info(f'network: {network}')
        dhlogger.info(f'num_envs_per_agent: {num_envs}')

        self.alg = alg
        self.network = network
        self.num_envs_per_agent = num_envs

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument("--problem",
                            type=str,
                            help="Problem to use for the search.",
                            default="deephyper.benchmark.nas.linearReg.Problem",
                            )
        parser.add_argument("--run",
                            type=str,
                            help="Run function to use for evaluations.",
                            default="deephyper.search.nas.model.run.quick",
                            )
        parser.add_argument('--max-evals',
                            type=int,
                            default=10,
                            help='maximum number of evaluations.')
        parser.add_argument('--network',
                            type=str,
                            default='ppo_lstm',
                            choices=['ppo_lstm'],
                            help='Policy-Value network.')
        return parser

    def main(self):

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])

        self.train(space=self.space,
                   evaluator=self.evaluator,
                   alg=self.alg,
                   network=self.network,
                   num_evals=self.num_evals,
                   num_envs=self.num_envs_per_agent)

    def train(self, space, evaluator, alg, network, num_evals, num_envs):
        """Function to train ours agents.

        Args:
            space (dict): space of the search (i.e. params dict)
            evaluator (Evaluator): evaluator we are using for the search.
            alg (str): TODO
            network (str): TODO
            num_evals (int): number of evaluations to run. (i.e. number of
                episodes for NAS)

        Returns:
            [type]: [description]
        """

        seed = 2019

        learn = get_learn_function(alg)
        alg_kwargs = get_learn_function_defaults(alg)
        for k in self.kwargs:
            if k in alg_kwargs:
                alg_kwargs[k] = self.kwargs[k]

        env = build_env(num_envs, space, evaluator)
        total_timesteps = num_evals * env.num_actions_per_env

        alg_kwargs['network'] = network
        alg_kwargs['nsteps'] = env.num_actions_per_env

        model = learn(
            env=env,
            seed=seed,
            total_timesteps=total_timesteps,
            **alg_kwargs
        )

        return model, env


def build_env(num_envs, space, evaluator):
    """Build nas environment.

    Args:
        num_envs (int): number of environments to run in parallel (>=1).
        space (dict): space of the search (i.e. params dict)
        evaluator (Evaluator): evaluator object to use.

    Returns:
        VecEnv: vectorized environment.
    """
    assert num_envs >= 1, f'num_envs={num_envs}'
    cs_kwargs = space['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = space['create_structure']['func']()
    else:
        structure = space['create_structure']['func'](**cs_kwargs)
    env = NeuralArchitectureVecEnv(num_envs, space, evaluator,
                                   structure)
    return env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = import_module(
        '.'.join(['deephyper.search.nas.baselines', alg, submodule]))
    try:
        # first try to import the alg module from baselines
        alg_module = import_module(
            '.'.join(['deephyper.search.nas.baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg):
    env_type = 'nas'
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


if __name__ == '__main__':
    kwargs = get_learn_function_defaults('ppo2')
    from pprint import pprint
    pprint(kwargs)

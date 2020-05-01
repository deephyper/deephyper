import json
import math
import os
import datetime
from importlib import import_module

import tensorflow as tf

from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch
from deephyper.evaluator.evaluate import Encoder
from deephyper.search.nas.baselines import logger
from deephyper.search.nas.envs import NasEnv1, NasEnv2
from deephyper.core.logs.logging import JsonMessage as jm

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


dhlogger = util.conf_logger("deephyper.search.nas.rl")


def get_env_class(env_name):
    """Return environment class corresponding to environment name"""
    envs = {"NasEnv1": NasEnv1, "NasEnv2": NasEnv2}
    assert env_name in envs.keys(), f"Environment '{env_name}' does not exist!"
    return envs[env_name]


class ReinforcementLearningSearch(NeuralArchitectureSearch):
    """Represents different kind of RL algorithms working with NAS.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
        alg (str): algorithm to use among ['ppo2',].
        network (str/function): policy network.
        cache_key (str): ...
    """

    def __init__(self, problem, run, evaluator, alg, network, env, **kwargs):

        self.kwargs = kwargs

        # It is assumed here that one agent is using one node
        self.num_agents = 1 if MPI is None else MPI.COMM_WORLD.Get_size()
        self.rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            super().__init__(
                problem=problem,
                run=run,
                evaluator=evaluator,
                num_nodes_master=self.num_agents,
                **kwargs,
            )  # create self.evaluator

        MPI.COMM_WORLD.Barrier()  # Creating the app first in rank 0 only

        if self.rank != 0:
            super().__init__(
                problem=problem,
                run=run,
                evaluator=evaluator,
                num_nodes_master=self.num_agents,
                **kwargs,
            )
        # Set in super : self.problem, self.run_func, self.evaluator

        # Number of parallel environments per agent
        self.num_envs_per_agent = self.evaluator.num_workers // self.num_agents

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            dhlogger.info(
                jm(
                    type="start_infos",
                    seed=self.problem.seed,
                    alg=alg,
                    network=network,
                    num_envs_per_agent=self.num_envs_per_agent,
                    nagents=1,
                    nworkers=self.evaluator.num_workers,
                    encoded_space=json.dumps(self.problem.space, cls=Encoder),
                )
            )

        if self.problem.seed is not None:
            tf.random.set_random_seed(self.problem.seed)

        self.num_evals = self.max_evals
        if self.num_evals is None:
            self.num_evals = math.inf

        self.space = self.problem.space

        dhlogger.info(f"evaluator: {type(self.evaluator)}")
        dhlogger.info(f"rank: {self.rank}")
        dhlogger.info(f"alg: {alg}")
        dhlogger.info(f"network: {network}")
        dhlogger.info(f"num_envs_per_agent: {self.num_envs_per_agent}")

        self.alg = alg
        self.network = network
        self.env_class = get_env_class(env)

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        return parser

    def main(self):

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])

        self.train(
            space=self.space,
            evaluator=self.evaluator,
            alg=self.alg,
            network=self.network,
            num_evals=self.num_evals + 1,
            num_envs=self.num_envs_per_agent,
        )

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

        seed = self.problem.seed

        learn = get_learn_function(alg)
        alg_kwargs = get_learn_function_defaults(alg)
        for k in self.kwargs:
            if k in alg_kwargs:
                alg_kwargs[k] = self.kwargs[k]

        env = build_env(num_envs, space, evaluator, self.env_class)
        total_timesteps = num_evals * env.num_actions_per_env

        alg_kwargs["network"] = network
        alg_kwargs["nsteps"] = env.num_actions_per_env

        model = learn(env=env, seed=seed, total_timesteps=total_timesteps, **alg_kwargs)

        return model, env


def build_env(num_envs, space, evaluator, env_class):
    """Build nas environment.

    Args:
        num_envs (int): number of environments to run in parallel (>=1).
        space (dict): space of the search (i.e. params dict)
        evaluator (Evaluator): evaluator object to use.

    Returns:
        VecEnv: vectorized environment.
    """
    assert num_envs >= 1, f"num_envs={num_envs}"
    cs_kwargs = space["create_search_space"].get("kwargs")
    if cs_kwargs is None:
        search_space = space["create_search_space"]["func"]()
    else:
        search_space = space["create_search_space"]["func"](**cs_kwargs)
    env = env_class(num_envs, space, evaluator, search_space)
    return env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = import_module(
        ".".join(["deephyper.search.nas.baselines", alg, submodule])
    )
    try:
        # first try to import the alg module from baselines
        alg_module = import_module(
            ".".join(["deephyper.search.nas.baselines", alg, submodule])
        )
    except ImportError:
        # then from rl_algs
        alg_module = import_module(".".join(["rl_" + "algs", alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg):
    env_type = "nas"
    try:
        alg_defaults = get_alg_module(alg, "defaults")
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

import json
import os.path as osp

import tensorflow as tf
from mpi4py import MPI

import deephyper.search.nas.utils.common.tf_util as U
from deephyper.evaluator import Evaluator
from deephyper.search.nas.agent import pposgd_async
from deephyper.search.nas.agent.policy import lstm
from deephyper.search.nas.env import NasEnv
from  deephyper.search.nas.utils.common import set_global_seeds


def train(num_episodes,
        seed,
        space,
        evaluator,
        num_episodes_per_batch,
        reward_rule,
        clip_param=0.2,
        entcoeff=0.01,
        optim_epochs=4,
        optim_stepsize=1e-3,
        optim_batchsize=None,
        gamma=0.99,
        lam=0.95):

    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    # if rank == 0:
    #     logger.configure()
    # else:
    #     logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    # MAKE ENV_NAS
    cs_kwargs = space['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = space['create_structure']['func']()
    else:
        structure = space['create_structure']['func'](**cs_kwargs)

    num_nodes = structure.num_nodes
    timesteps_per_actorbatch = num_nodes * num_episodes_per_batch
    num_timesteps = timesteps_per_actorbatch * num_episodes
    # print('num_nodes: ', num_nodes)
    # print('timesteps_per_actorbatch: ', timesteps_per_actorbatch)
    # print('num_episodes_per_batch: ', num_episodes_per_batch)

    if optim_batchsize is None:
        optim_batchsize = structure.num_nodes

    env = NasEnv(space, evaluator, structure)

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return lstm.LstmPolicy(
            name=name,
            ob_space=ob_space,
            ac_space=ac_space,
            num_units=32,
            async_update=True)

    pposgd_async.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=clip_param,
        entcoeff=entcoeff,
        optim_epochs=optim_epochs,
        optim_stepsize=optim_stepsize,
        optim_batchsize=optim_batchsize,
        gamma=gamma,
        lam=lam,
        schedule='linear',
        reward_rule=reward_rule
    )

    env.close()

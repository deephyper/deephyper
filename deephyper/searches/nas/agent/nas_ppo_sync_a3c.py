import json
import os.path as osp

import tensorflow as tf
from mpi4py import MPI

import deephyper.searches.nas.utils.common.tf_util as U
from deephyper.evaluators import Evaluator
from deephyper.searches.nas.agent import pposgd_sync
from deephyper.searches.nas.agent.policy import lstm
from deephyper.searches.nas.envs import NasEnv
from  deephyper.searches.nas.utils import bench, logger
from  deephyper.searches.nas.utils.common import set_global_seeds


def train(num_episodes, seed, space, evaluator, num_episodes_per_batch, reward_rule):

    rank = MPI.COMM_WORLD.Get_rank()
    env_id = rank
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    # MAKE ENV_NAS
    structure = space['create_structure']['func'](
        tf.constant([[1., 1.]]),
        **space['create_structure']['kwargs']
    )

    num_nodes = structure.num_nodes
    timesteps_per_actorbatch = num_nodes * num_episodes_per_batch
    num_timesteps = timesteps_per_actorbatch * num_episodes

    env = NasEnv(space, evaluator, structure)

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return lstm.LstmPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            num_units=32)


    pposgd_sync.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        entcoeff=0.01,
        optim_epochs=4,
        optim_stepsize=1e-3,
        optim_batchsize=15,
        gamma=0.99,
        lam=0.95,
        schedule='linear',
        reward_rule=reward_rule
    )
    env.close()

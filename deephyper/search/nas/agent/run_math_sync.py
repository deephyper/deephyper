import json
import os.path as osp

import tensorflow as tf
from mpi4py import MPI

import deephyper.search.nas.utils.common.tf_util as U
from deephyper.evaluator import Evaluator
from deephyper.search.nas.agent import pposgd_sync_ph
from deephyper.search.nas.agent.policy import lstm_ph 
from deephyper.search.nas.env import MathEnv
from deephyper.search.nas.utils import bench, logger
from deephyper.search.nas.utils.common import set_global_seeds
from deephyper.search.nas.agent.utils import reward_for_final_timestep


def train(num_iter, seed, evaluator, num_episodes_per_iter):

    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    # MAKE ENV_NAS
    timesteps_per_episode = 10
    timesteps_per_actorbatch = timesteps_per_episode*num_episodes_per_iter
    num_timesteps = timesteps_per_actorbatch * num_iter

    env = MathEnv(evaluator)

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return lstm_ph.LstmPolicy(name=name, ob_space=ob_space, ac_space=ac_space, num_units=64)

    pposgd_sync_ph.learn(env, policy_fn,
        max_timesteps=int(num_timesteps),
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        entcoeff=0.01, #0.01,
        optim_epochs=4,
        optim_stepsize=1e-3,
        optim_batchsize=10,
        gamma=0.99, # 0.99
        lam=0.95, # 0.95
        schedule='linear',
        reward_rule=reward_for_final_timestep
    )
    env.close()

def key(d):
    return json.dumps(dict(arch_seq=d['arch_seq']))

def main():
    from deephyper.search.nas.agent.run_func_math import run_func
    evaluator = Evaluator.create(run_func, cache_key=key, method='threadPool')
    train(
        num_iter=500,
        num_episodes_per_iter=10,
        seed=2018,
        evaluator=evaluator)

if __name__ == '__main__':
    main()

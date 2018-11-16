import json
import os.path as osp

import numpy as np
import tensorflow as tf
from mpi4py import MPI

import deephyper.search.nas.utils.common.tf_util as U
from deephyper.evaluator import Evaluator
from deephyper.search.nas.env import NasEnv
from deephyper.search.nas.utils import bench, logger
from deephyper.search.nas.utils.common import set_global_seeds
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm

dh_logger = util.conf_logger('deephyper.search.nas.agent.nas_random')


def traj_segment_generator(env, horizon):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    ts_i2n_ep = {}

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    num_evals = 0

    while True:
        prevac = ac
        ac = env.action_space.sample()
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            while num_evals > 0:
                results = env.get_rewards_ready()
                for (cfg, rew) in results:
                    index = cfg['w']
                    rews[index] = rew
                    num_evals -= 1
                    ep_rets[ts_i2n_ep[index]-1] += rew
            ts_i2n_ep = {}
            data = {"ob" : obs, "rew" : rews, "new" : news,
                    "ac" : acs, "prevac" : prevacs,
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            yield data
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # observ, reward, episode_over, meta -> {}
        ob, rew, new, _ = env.step(ac, i, rank=MPI.COMM_WORLD.Get_rank())
        rews[i] = rew

        cur_ep_ret += rew if rew != None else 0
        cur_ep_len += 1
        if new:
            num_evals += 1
            ts_i2n_ep[i] =  num_evals
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def train(num_episodes, seed, space, evaluator, num_episodes_per_batch):

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: # rank zero simule the use of a parameter server
        pass
    else:
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
        set_global_seeds(workerseed)

        # MAKE ENV_NAS
        structure = space['create_structure']['func'](
            **space['create_structure']['kwargs']
        )

        num_nodes = structure.num_nodes
        timesteps_per_actorbatch = num_nodes * num_episodes_per_batch
        num_timesteps = timesteps_per_actorbatch * num_episodes

        max_timesteps = num_timesteps
        timesteps_per_actorbatch=timesteps_per_actorbatch

        env = NasEnv(space, evaluator, structure)

        seg_gen = traj_segment_generator(env, timesteps_per_actorbatch)

        timesteps_so_far = 0
        iters_so_far = 0

        cond = sum([max_timesteps>0])
        assert cond==1, f"Only one time constraint permitted: cond={cond}, max_timesteps={max_timesteps}"

        while True:
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break

            logger.log("********** Iteration %i ************"%iters_so_far)

            seg = seg_gen.__next__()
            dh_logger.info(jm(type='seg', rank=MPI.COMM_WORLD.Get_rank(), **seg))
            iters_so_far += 1

        env.close()

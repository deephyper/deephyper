import numpy as np

from mpi4py import MPI

def final_reward_for_all_timesteps(reward_list, index_final_timestep, reward, episode_length):
    """
    Args:
        reward_list (list): list of length (episode_length) * number_of_episodes
        index_final_timestep (int): index of final timestep of current episode in reward_list
        reward (float): reward corresponding to the current episode
        episode_length (int): length of the current episode

    Return:
        Reward of current episode.
    """
    episode_reward = reward * episode_length
    for i in range(0, episode_length):
        reward_list[index_final_timestep-i] = reward
    return episode_reward

def episode_reward_for_final_timestep(reward_list, index_final_timestep, reward, episode_length):
    """
    Args:
        reward_list (list): list of length (episode_length) * number_of_episodes
        index_final_timestep (int): index of final timestep of current episode in reward_list
        reward (float): reward corresponding to the current episode
        episode_length (int): length of the current episode

    Return:
        Reward of current episode.
    """
    episode_reward = reward
    for i in range(1, episode_length):
        reward_list[index_final_timestep-i] = 0
    reward_list[index_final_timestep] = reward
    return episode_reward


def traj_segment_generator(pi, env, horizon, stochastic, reward_affect_func):
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
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    num_evals = 0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            while num_evals > 0:
                results = env.get_rewards_ready()
                for (cfg, rew) in results:
                    index = cfg['w']
                    episode_length = ep_lens[ts_i2n_ep[index]-1]
                    episode_rew = reward_affect_func(rews, index, rew, episode_length)
                    num_evals -= 1
                    ep_rets[ts_i2n_ep[index]-1] = episode_rew
            ts_i2n_ep = {}
            data = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            yield data
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
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


def traj_segment_generator_ph(pi, env, horizon, stochastic, reward_affect_func):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    c_vf, h_vf = np.zeros([1]+list(pi.input_c_vf.get_shape()[1:])), np.zeros([1]+list(pi.input_h_vf.get_shape()[1:]))
    c_pol, h_pol = np.zeros([1]+list(pi.input_c_pol.get_shape()[1:])), np.zeros([1]+list(pi.input_h_pol.get_shape()[1:]))

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    ts_i2n_ep = {}

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    history_hs_vf = [None for _ in range(horizon)]
    history_hs_pol = [None for _ in range(horizon)]
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    num_evals = 0

    while True:
        prevac = ac
        ac, vpred, (c_vf, h_vf), (c_pol, h_pol) = pi.act(stochastic, ob, c_vf, h_vf, c_pol, h_pol)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            while num_evals > 0:
                results = env.get_rewards_ready()
                for (cfg, rew) in results:
                    index = cfg['w']
                    episode_length = ep_lens[ts_i2n_ep[index]-1]
                    episode_rew = reward_affect_func(rews, index, rew, episode_length)
                    num_evals -= 1
                    ep_rets[ts_i2n_ep[index]-1] = episode_rew
            ts_i2n_ep = {}
            data = {
                "ob" : obs, 
                "hs_vf": history_hs_vf,
                "hs_pol": history_hs_pol,
                "rew" : rews, 
                "vpred" : vpreds, 
                "new" : news,
                "ac" : acs, 
                "prevac" : prevacs, 
                "nextvpred": vpred * (1 - new),
                "ep_rets" : ep_rets, 
                "ep_lens" : ep_lens
            }
            yield data
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        history_hs_vf[i] = (c_vf, h_vf)
        history_hs_pol[i] = (c_pol, h_pol)
        vpreds[i] = vpred
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
            c_vf, h_vf = np.zeros([1]+list(pi.input_c_vf.get_shape()[1:])), np.zeros([1]+list(pi.input_h_vf.get_shape()[1:]))
            c_pol, h_pol = np.zeros([1]+list(pi.input_c_pol.get_shape()[1:])), np.zeros([1]+list(pi.input_h_pol.get_shape()[1:]))
        t += 1

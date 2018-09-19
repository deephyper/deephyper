import os
import sys
import tensorflow as tf
import numpy as np
import seaborn as sns
import random
import math
import logging
import pprint
from io import BytesIO
tf.set_random_seed(1000003)
np.random.seed(1000003)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV5
from deephyper.search.nas.reinforce.tf import BasicReinforceV5
from deephyper.model.arch import StateSpace

from benchmark_functions_wrappers import *

def equals(v, length=10):
    if (len(v) <= length):
        return False
    b = True
    length = length if len(v) >= length else len(v)
    for i in range(1, length):
        b = b and (v[-i] == v[-(i+1)])
    return b

def create_set_directory(n, func, params):
    directory = f'sample.{n}_fc.{func.__name__}'
    for key in sorted(params.keys()):
        key_split = key.split('_')
        fst = [s[0] for s in key_split]
        directory += f'_{"".join(fst)}.{params[key]}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chdir(directory)
    return directory

def get_logger(func, step):
    logger = logging.getLogger(f'{func.__name__}_{step}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{func.__name__}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def test_BasicReinforceV5_PPO_batch(step, func, batch_size=1, algo='RANDOM'):
    '''
    Test for Proximal Policy Gradient reinforcement with different batch_size.
    '''
    tf.reset_default_graph()
    # parameters
    params = {
        'learning_rate': 0.0006,
        'max_layers': 1,
        'algo': algo,
        'clip_param': 0.2,
        'entropy_param': 0.0,
        'vf_loss_param': 0.,
        'discount_factor': 0.99,
        'batch_size': batch_size,
        'max_iter': 1000,
        'num_val': 10,
        'num_dim': 10,
        'num_units': 10,
        'method': 'SEQ', # 'SEQ' or 'BATCH'
        'batch_size_method': 1 } # only for method: 'BATCH'
    learning_rate = params['learning_rate']
    max_layers = params['max_layers']
    algo = params['algo']
    clip_param = params['clip_param']
    entropy_param = params['entropy_param']
    batch_size = params['batch_size']
    max_iter = params['max_iter']
    num_val = params['num_val']
    num_dim = params['num_dim']
    num_units = params['num_units']
    method = params['method']
    batch_size_method = params['batch_size_method']
    discount_factor = params['discount_factor']
    vf_loss_param = params['vf_loss_param']

    directory = create_set_directory(step, func, params)
    logger = get_logger(func, step)
    logger.debug(f'Experiment save in directory : {directory}')
    logger.debug(f'parameters: {pprint.pformat(params)}')
    func, (a, b), _ = func()

    # Tensorflow session
    session = tf.Session()

    # StateSpace creation
    state_space = StateSpace()
    x_values = np.linspace(a, b, num_val)
    for i in range(num_dim):
        state_space.add_state(f'x_{i+1}', x_values)

    if algo != 'RANDOM':
        # Policy Network creation
        policy_network = NASCellPolicyV5(state_space, num_units=num_units)

        # optimizer for Reinforce object
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Reinforce object creation
        global_step = tf.Variable(0, trainable=False)
        reinforce = BasicReinforceV5(session,
                                    optimizer,
                                    policy_network,
                                    max_layers,
                                    batch_size, #async
                                    global_step,
                                    optimization_algorithm=algo,
                                    clip_param=clip_param,
                                    entropy_param=entropy_param,
                                    vf_loss_param=vf_loss_param,
                                    discount_factor=discount_factor,
                                    state_space=state_space)

    # Init
    seeds = [ 1 for i in range(batch_size)]
    actions_history = []
    rewards_history = []


    # method = 'SEQ'
    if method == 'SEQ' or algo == 'RANDOM':
        for i in range(max_iter):
            logger.debug(f'\niter       = {i}')
            if algo != 'RANDOM':
                # Generating a 'pre'-action (index of tokens)
                actions, _ = reinforce.get_actions(seeds,
                                            max_layers)
            for n in range(batch_size):
                if algo != 'RANDOM':
                    action = actions[n:len(actions):batch_size]
                    logger.debug(f'pre-action = {action}')

                    # Converting indexes to values
                    conv_action = state_space.parse_state(action, num_layers=max_layers)
                else:
                    conv_action = state_space.get_random_state_space(1)[0]
                actions_history.append(conv_action)
                logger.debug(f'action     = {conv_action}')

                # Generation the reward from the previous action
                reward = func(conv_action)
                rewards = [reward]
                rewards_history.append(reward)
                logger.debug(f'reward     = {reward}')

            if algo != 'RANDOM':
                reinforce.storeRollout(actions, rewards, max_layers)
                reinforce.train_step(max_layers, seeds, i)

            if equals(rewards_history):
                break
    elif method == 'BATCH':
        for i in range(max_iter//batch_size_method):
            logger.debug(f'\niter       = {i*batch_size_method}')
            # Generating actions
            action_list = []
            for action_n in range(batch_size_method):
                # pre_action is list of indexes corresponding to tokens
                pre_actions, _ = reinforce.get_actions(seeds, max_layers)
                pre_action = pre_actions[0:len(pre_actions):batch_size]
                # action is list of tokens
                action = state_space.parse_state(pre_action, num_layers=max_layers)
                action_list.append(action)

                actions_history.append(action)
                logger.debug(f'action_{action_n} = {action}')

            # Getting rewards
            for reward_n in range(batch_size_method):
                action = action_list[reward_n]
                reward = func(action)
                rewards = [reward]

                rewards_history.append(reward)
                logger.debug(f'reward = {reward}')

                reinforce.storeRollout(pre_actions, rewards, max_layers)
                reinforce.train_step(max_layers, seeds, i)

            if equals(rewards_history):
                break
    else:
        print(f'Selected method: "{method}" doesn\'t exist !')

    session.close()

    return actions_history, rewards_history

def max_len(m):
    mx = 0
    for l in m:
        mx = max(len(l), mx)
    return mx

def complete(l, max_len):
    while (len(l) < max_len):
        l.append(l[-1])

def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

def do_box_plot(step, algo, historyAlgo, historyRandom):
    # Algo plot
    history = historyAlgo
    rewards = []
    mx = 0
    for _, r in history:
        rewards.append(r[:])
        mx = max(mx, len(r))
    for l in rewards:
        complete(l, mx)
    # v1, (min_y, max_y) = max_mean_min(rewards)
    v2, (min_y, max_y) = raw(rewards) #raw(rewards)
    # min_y, max_y = min(min_y, mi), max(max_y, ma)
    v3, (mi, ma) = max_rewards(rewards)
    min_y, max_y = min(min_y, mi), max(max_y, ma)
    # plot_box('Max.mean.min', v1, min_y, max_y)
    plt.subplot(221)
    plot_box(f'Raw iterations : {algo}', v2, min_y, max_y)
    plt.xlim(xmin=0)
    plt.ylim(ymax=0)
    plt.subplot(223)
    plot_box(f'Max rewards over time : {algo}', v3, min_y, max_y)
    plt.annotate(str(v3[1][-1])[:7],xy=(v3[0][-1], v3[1][-1]))
    plt.xlim(xmin=0)
    plt.ylim(ymax=0)
    _, xmax = plt.xlim()

    # Random plot
    history = historyRandom
    rewards = []
    mx = 0
    for _, r in history:
        rewards.append(r[:])
        mx = max(mx, len(r))
    for l in rewards:
        complete(l, mx)
    # v1, (min_y, max_y) = max_mean_min(rewards)
    v2, (min_y, max_y) = raw(rewards)
    # min_y, max_y = min(min_y, mi), max(max_y, ma)
    v3, (mi, ma) = max_rewards(rewards)
    min_y, max_y = min(min_y, mi), max(max_y, ma)
    # plot_box('Max.mean.min', v1, min_y, max_y
    plt.subplot(222)
    plot_box('Raw iterations : RANDOM', v2, min_y, max_y)
    plt.xlim(0, xmax)
    plt.ylim(ymax=0)
    plt.subplot(224)
    plot_box('Max rewards over time : RANDOM', v3, min_y, max_y)
    plt.annotate(str(v3[1][-1])[:7], xy=(v3[0][-1], v3[1][-1]))
    plt.xlim(0, xmax)
    plt.ylim(ymax=0)

    save_plot(f'{step}_step')

def max_mean_min(rewards):
    arr = np.array(rewards)
    len_row = np.shape(arr)[1]
    mean = np.mean(rewards, axis=0)
    min_mean = []
    for i in range(len_row):
        min_mean.append(min(arr[:, i]))
    max_mean = []
    for i in range(len_row):
        max_mean.append(max(arr[:, i]))
    x = [i for i in range(len(mean))]
    return (x, mean, min_mean, max_mean), (min(min_mean), max(max_mean))

def raw(rewards):
    # raw iterations
    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    down_mean = mean - std
    up_mean = mean + std
    x = [i for i in range(len(mean))]
    # return plot_box('Raw.iterations', mean, std)
    y_min, y_max = min(down_mean), max(up_mean)
    return (x, mean, down_mean, up_mean), (y_min, y_max)

def max_rewards(rewards):
    # max over time
    mx_rewards = []
    for l in rewards:
        mx_rewards.append(max_list(l))
    mean = np.mean(mx_rewards, axis=0)
    std = np.std(mx_rewards, axis=0)
    down_mean = mean - std
    up_mean = mean + std
    x = [i for i in range(len(mean))]
    y_min, y_max = min(down_mean), max(up_mean)
    return (x, mean, down_mean, up_mean), (y_min, y_max)

def plot_box(prefix, values, y_min, y_max):
    (x, mean, down_mean, up_mean) = values
    name = f'{prefix}' #_{len(history)}.sample(s)'
    plt.title(name)
    plt.ylim(y_min, y_max)
    plt.plot(x, up_mean, 'r:')
    plt.plot(x, mean, 'b')
    plt.plot(x, down_mean, 'r:')
    plt.fill_between(x, mean, up_mean, facecolor='cyan', alpha=0.5)
    plt.fill_between(x, mean, down_mean, facecolor='cyan', alpha=0.5)

def save_plot(name):
    plt.savefig(name+'.png', dpi=200)
    plt.savefig(name+'.svg')
    plt.clf()

if __name__ == '__main__':
    # funcs = [polynome_2, ackley_, dixonprice_, griewank_, levy_]
    # funcs = [polynome_2, ackley_]
    # funcs = [dixonprice_, griewank_, levy_]
    # funcs = [polynome_2]
    funcs = [ackley_]
    # funcs = [dixonprice_]
    # funcs = [griewank_]
    # funcs = [levy_]
    batch_size = 1
    num_samples = 10

    main_cwd = os.getcwd()
    for func in funcs:
        os.chdir(main_cwd)
        historyAlgo, historyRandom = [], []
        directory = func.__name__
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)
        local_cwd = os.getcwd()
        for step in range(num_samples):

            # Algo
            os.chdir(local_cwd)
            algo = 'PPO'
            # algo = 'PG'
            acts, rwds = test_BasicReinforceV5_PPO_batch(step, func, batch_size, algo)
            historyAlgo.append((acts, rwds))

            # Random
            os.chdir(local_cwd)
            acts, rwds = test_BasicReinforceV5_PPO_batch(step, func, batch_size, 'RANDOM')
            historyRandom.append((acts, rwds))

            os.chdir(local_cwd)
            plt.figure(figsize=(16, 10), dpi=300, facecolor='w', edgecolor='k')
            do_box_plot(step, algo, historyAlgo, historyRandom)
            tf.get_variable_scope().reuse_variables()

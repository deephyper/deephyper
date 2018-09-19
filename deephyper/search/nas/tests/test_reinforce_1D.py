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
import matplotlib.animation as animation

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV5
from deephyper.search.nas.reinforce.tf import BasicReinforceV5
from deephyper.model.arch import StateSpace

from benchmark_functions_wrappers import *

def equals(v, length=10):
    if (1 >= len(v)):
        return False
    b = True
    length = length if len(v) >= length else len(v)
    for i in range(1, length):
        b = b and (v[-i] == v[-(i+1)])
    return b

def create_set_directory(func, params):
    directory = f'fc.{func.__name__}'
    for key in sorted(params.keys()):
        key_split = key.split('_')
        fst = [s[0] for s in key_split]
        directory += f'_{"".join(fst)}.{params[key]}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chdir(directory)
    return directory

def get_logger(func):
    logger = logging.getLogger(f'{func.__name__}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{func.__name__}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def test_BasicReinforceV5_PPO_batch(func, batch_size=1):
    '''
    Test for Proximal Policy Gradient reinforcement with different batch_size.
    '''
    # parameters
    params = {
        'learning_rate': 1.,
        'max_layers': 1,
        'algo': 'PPO',
        'clip_param': 0.1,
        'entropy_param': 0.,
        'batch_size': batch_size,
        'max_iter': 2000,
        'num_val': 20,
        'num_units': 2 }
    learning_rate = params['learning_rate']
    max_layers = params['max_layers']
    algo = params['algo']
    clip_param = params['clip_param']
    entropy_param = params['entropy_param']
    batch_size = params['batch_size']
    max_iter = params['max_iter']
    num_val = params['num_val']
    num_units = params['num_units']

    directory = create_set_directory(func, params)
    logger = get_logger(func)
    logger.debug(f'Experiment save in directory : {directory}')
    logger.debug(f'parameters: {pprint.pformat(params)}')
    func, (a, b), minima = func()
    minima = minima(1)

    # Tensorflow session
    session = tf.Session()

    # StateSpace creation
    state_space = StateSpace()
    x_values = np.linspace(a, b, num_val)
    state_space.add_state('x', x_values)

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
                                state_space=state_space)

    # Init
    seeds = [ 1 for i in range(batch_size)]
    actions_history = []
    rewards_history = []
    session.run(tf.global_variables_initializer())

    it = 0
    for i in range(max_iter):
        it += 1
        logger.debug(f'\niter       = {i}')
        # Generating a 'pre'-action (index of tokens)
        actions, _ = reinforce.get_actions(seeds,
                                        max_layers)
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            logger.debug(f'pre-action = {action}')

            # Converting indexes to values
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            actions_history.append(conv_action[0])
            logger.debug(f'action     = {conv_action[0]}')

            # Generation the reward from the previous action
            reward = func(conv_action)
            rewards = [reward]
            rewards_history.append(reward)
            logger.debug(f'reward     = {reward}')

        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, seeds, i)

        if equals(rewards_history):
            break

    plt.title(directory)
    plt.plot(x_values, [func([x]) for x in x_values], 'b')
    color_map = cm.get_cmap('hot')([i for i in range(it)])
    color_map_iter = iter(color_map)
    for i in range(it):
        plt.scatter(actions_history[i*batch_size:(i+1)*batch_size],
                    rewards_history[i*batch_size:(i+1)*batch_size],
                    c=next(color_map_iter),)
    plt.plot(actions_history[-1], rewards_history[-1], 'bo') # last action
    # plt.colorbar()
    plt.savefig('fig.svg')
    plt.savefig('fig.png')
    plt.clf()
    session.close()


if __name__ == '__main__':
    funcs = [polynome_2, ackley_, dixonprice_, griewank_, levy_]
    funcs = [levy_]
    batch_size = 1

    cwd = os.getcwd()
    for func in funcs:
        os.chdir(cwd)
        test_BasicReinforceV5_PPO_batch(func, batch_size)
        break
        tf.get_variable_scope().reuse_variables()

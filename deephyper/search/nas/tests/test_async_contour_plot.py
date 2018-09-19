import os
import sys
import tensorflow as tf
import numpy as np
import seaborn as sns
import random
import math
import logging
from io import BytesIO
tf.set_random_seed(1000003)
np.random.seed(1000003)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV5
from deephyper.search.nas.reinforce.tf import BasicReinforceV5
from deephyper.model.arch import StateSpace

from benchmark_functions import *

NB_ITER = 4000
NUM_VAL = 100
NUM_DIM = 2

def get_seeds_uniform(x):
    return [int(abs(float(np.random.uniform(0,1))) * NUM_VAL) - NUM_VAL//2 for i in range(x)]

def equals(v, length=10):
    if (1 >= len(v)):
        return False
    b = True
    length = length if len(v) >= length else len(v)
    for i in range(1, length):
        b = b and (v[-i] == v[-(i+1)])
    return b

def get_seeds_normal(x):
    return [int(np.random.normal() * NUM_VAL)//2 for i in range(x)]

BATCH_SIZE = 1
RANDOM_SEEDS = get_seeds_uniform(BATCH_SIZE)
# init_seeds = RANDOM_SEEDS
init_seeds = [1]

def test_fixed_num_layers(f):
    func, I, minimas = f()

    logger = logging.getLogger(f'{f.__name__}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{f.__name__}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    num_val = NUM_VAL
    a, b = I
    minimas = minimas(NUM_DIM)

    values = np.linspace(a, b, num_val)
    x1_values = values
    state_space.add_state('x1', x1_values)
    x2_values = values
    state_space.add_state('x2', x2_values)

    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    z_mesh = np.zeros((NUM_VAL, NUM_VAL))
    for i in range(NUM_VAL):
        for j in range(NUM_VAL):
            z_mesh[i,j] = func([x1_mesh[i,j], x2_mesh[i,j]])

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = BATCH_SIZE

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    # learning_rate = 0.0006
    learning_rate = 0.0001

    # optimizer = tf.train.RMSPopOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                1, #async
                                global_step,
                                state_space=state_space)

    tf.summary.FileWriter('graph', graph=tf.get_default_graph())

    lx3 = [] # points
    rewards_history = []

    for num in range(1, NB_ITER+1):
        # init_seeds = get_seeds_normal(batch_size)
        num = num+1 # offset of one because log scale on axis x
        logger.debug(f'init_seeds: {init_seeds}')

        actions = []
        for b in range(batch_size):
            action, _ = reinforce.get_actions([init_seeds[0]], max_layers)
            actions.append(action)
        random.shuffle(actions)
        rewards = []

        for n in range(batch_size):
            action = actions[n]
            # print(f'action: {action}')
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            # conv_action = state_space.get_random_state_space(BATCH_SIZE)[0]
            reward = func(conv_action)
            rewards.append(reward)
            rewards_history.append(reward)
            logger.debug(f'STEP = {num*batch_size + n}')
            try:
                logger.debug(f'reward: {reward} max_rewards: {reinforce.max_reward} ema: {reinforce.rewards_b} (R-b): {reinforce.R_b}')
                logger.debug(f'action: {action}')
                logger.debug(f'state_space: {conv_action}')
                logger.debug(f'minimas: {minimas}')
            except:
                pass

            reinforce.storeRollout(action, [reward], max_layers)
            reinforce.train_step(max_layers, [init_seeds[0]])

            lx3.append(conv_action)

        if equals(rewards_history):
            break

    lx3 = np.array(lx3)

    # v = [50, 100, 200, 400, 800, 1000]
    # v = [100*i for i in range(1, NB_ITER//100 + 1)]
    # v = [100]
    ###
    len_lx3 = len(lx3[:,0])
    size_slice = 100
    prev_num_points = 0
    num_points = size_slice
    while True:
        if num_points > len_lx3:
            num_points = len_lx3
        ttle = f'{f.__name__}, {NUM_VAL} values, iter = {num_points}, rnn policy gradient'
        print(ttle)
        plt.title(ttle)
        # plt.title(f'{f.__name__}, {NUM_VAL} values, iter = {num_points}, random')

        plt.contourf(x1_mesh, x2_mesh, z_mesh, 20, cmap='RdGy')
        plt.colorbar()

        plt.plot(minimas[0], minimas[1], 'co')

        # cp = sns.color_palette("Blues", NB_ITER)
        plt.scatter(lx3[prev_num_points:num_points, 0],
                    lx3[prev_num_points:num_points, 1],
            c=np.array([i for i in range(prev_num_points, num_points)]),
            cmap='hot', vmin=1, vmax=len_lx3)
        plt.colorbar()

        if num_points == len_lx3:
            plt.plot(lx3[-1, 0],
                    lx3[-1, 1], 'bo')

        with open(f'fig_{num_points}.png', 'wb') as file:
            plt.savefig(file, format="png")
        prev_num_points = num_points
        plt.clf()
        if num_points == len_lx3:
            break
        else:
            num_points += size_slice



def test_scheduled_num_layers(func):
    pass

def add(v):
    return -sum(v)

def powell_(v):
    return -powell(v)

def ackley_():
    '''
    Many local minimas
    global minimum = [0, 0, 0...0]
    '''
    max_ackley = lambda v: -ackley(v)
    a = -32.768
    b = 32.768
    minimas = lambda d: [0 for i in range(d)]
    return max_ackley, (a, b), minimas

def dixonprice_():
    '''
    Boal function
    global minimum = math.inf
    '''
    max_dixonprice = lambda v : -dixonprice(v)
    a = -10
    b = 10
    min_i = lambda i: 2**(-(2**i - 2)/(2**i))
    minimas = lambda d: [min_i(i) for i in range(d)]
    return max_dixonprice, (a, b), minimas

def polynome_2():
    p = lambda x: -sum([x_i**2 for x_i in x])
    a = -2
    b = 2
    minimas = lambda d: [0 for i in range(d)]
    return p, (a, b), minimas


if __name__ == '__main__':
    # f = polynome_2
    # f = ackley_
    f = dixonprice_
    test_fixed_num_layers(f)

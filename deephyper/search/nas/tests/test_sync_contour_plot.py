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
from benchmark_functions_wrappers import *

NB_ITER = 3000
NUM_VAL = 20
NUM_DIM = 2
BATCH_SIZE = 1
LEARNING_RATE = 1.

SLICE_SIZE = 100

ALGO = 'PPO'
CLIP_PARAM = 0.3
ENTROPY_PARAM = 0.

FUNCTION = polynome_2
# FUNCTION = ackley_
# FUNCTION = dixonprice_
# FUNCTION = griewank_
# FUNCTION = levy_

directory = f'fc.{FUNCTION.__name__}_bs.{BATCH_SIZE}_lr.{LEARNING_RATE}_cp.{CLIP_PARAM}_ep.{ENTROPY_PARAM}'
if not os.path.exists(directory):
    os.makedirs(directory)
    os.chdir(directory)

def get_seeds_uniform(x):
    return [int(-2 + np.random.uniform(0,1)*4) for i in range(x)]

def equals(v, length=10):
    if (1 >= len(v)):
        return False
    b = True
    length = length if len(v) >= length else len(v)
    for i in range(1, length):
        b = b and (v[-i] == v[-(i+1)])
    return b

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
    x2_values = values[::-1]
    state_space.add_state('x2', x2_values)

    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    z_mesh = np.zeros((NUM_VAL, NUM_VAL))
    for i in range(NUM_VAL):
        for j in range(NUM_VAL):
            z_mesh[i,j] = func([x1_mesh[i,j], x2_mesh[i,j]])

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = BATCH_SIZE

    # learning_rate = tf.train.exponential_decay(0.99, global_step,
    #                                             500, 0.96, staircase=True)

    learning_rate = LEARNING_RATE

    # optimizer = tf.train.RMSPopOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                BATCH_SIZE, #async
                                global_step,
                                optimization_algorithm=ALGO,
                                clip_param=CLIP_PARAM,
                                entropy_param=ENTROPY_PARAM,
                                state_space=state_space)

    tf.summary.FileWriter('graph', graph=tf.get_default_graph())
    seeds = [ 1 for i in range(batch_size)]
    lx3 = [] # points
    rewards_history = []

    for num in range(1, NB_ITER+1):
        # init_seeds = get_seeds_uniform(BATCH_SIZE)
        num = num+1 # offset of one because log scale on axis x
        logger.debug(f'init_seeds: {init_seeds}')

        actions, _ = reinforce.get_actions(seeds,
                                           max_layers)
        rewards = []
        try:
            for n in range(batch_size):
                action = actions[n:len(actions):batch_size]
                # action = action[::-1]
                # print(f'action: {action}')
                print(f'action: {action}')
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
                lx3.append(conv_action)
        except:
            break

        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, seeds)


        if equals(rewards_history):
            break

    lx3 = np.array(lx3)

    # v = [50, 100, 200, 400, 800, 1000]
    # v = [100*i for i in range(1, NB_ITER//100 + 1)]
    # v = [100]
    ###
    len_lx3 = len(lx3[:,0])
    size_slice = batch_size * SLICE_SIZE
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

        plt.plot(x1_mesh, x2_mesh, 'g+')

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

if __name__ == '__main__':
    f = FUNCTION
    test_fixed_num_layers(f)

import os
import sys
import tensorflow as tf
import numpy as np
import random
import math
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

NUM_VAL = 100
NUM_DIM = 2

def get_seeds_uniform(x):
    return [int(abs(float(np.random.uniform(0,1))) * NUM_VAL) - NUM_VAL//2 for i in range(x)]

def get_seeds_normal(x):
    return [int(np.random.normal() * NUM_VAL)//2 for i in range(x)]

BATCH_SIZE = 1
RANDOM_SEEDS = get_seeds_uniform(BATCH_SIZE)
# init_seeds = RANDOM_SEEDS
init_seeds = [1]

def test_fixed_num_layers(func, I, minimas):
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    num_val = NUM_VAL
    a, b = I
    minimas = minimas(NUM_DIM)

    values = np.linspace(a, b, num_val)
    # values.reverse()
    for d in range(NUM_DIM):
        # state_space.add_state('x%d' % d, values+(1*d))
        state_space.add_state('x%d' % d, values)

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = BATCH_SIZE

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    # learning_rate = 0.0006
    # learning_rate = 0.0001
    learning_rate = 1.0

    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                1, #async
                                global_step,
                                state_space=state_space)

    # init_seeds = RANDOM_SEEDS

    def update_line(num, line1, line2, minimas):
        global init_seeds, prev_rewards
        # init_seeds = get_seeds_normal(batch_size)
        num = num+1 # offset of one because log scale on axis x
        print(f'init_seeds: {init_seeds}')

        actions = []
        for b in range(batch_size):
            action, so = reinforce.get_actions([init_seeds[b]], max_layers)
            actions.append(action)
        random.shuffle(actions)
        rewards = []
        prev_rewards = rewards

        lx1, ly1 = line1.get_data()
        lx2, ly2 = line2.get_data()

        for n in range(batch_size):
            action = actions[n]
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            # conv_action = state_space.get_random_state_space(BATCH_SIZE)[0]
            reward = func(conv_action)
            rewards.append(reward)
            try:
                print(f'STEP = {num*batch_size + n} reward: {reward} max_rewards: {reinforce.max_reward} ema: {reinforce.rewards_b} (R-b): {reinforce.R_b}')
                print(f'prob: {so}')
                print(f'action: {action}')
                print(f'state_space: {conv_action}')
                print(f'minimas: {minimas}')
                #print(f'STEP = {num} actions: {actions} exp: {reinforce.exploration} rewards: {rewards} max_rewards: {reinforce.max_reward} ema: {reinforce.rewards_b}')
            except:
                pass

            reinforce.storeRollout(action, [reward], max_layers)
            reinforce.train_step(max_layers, [init_seeds[n]])
            # init_seeds = [action[0] for action in actions]
            # init_seeds = [np.linalg.norm(conv_action) for action in actions]

            ly1 = np.append(ly1, [reward])
            ly2 = np.append(ly2, [reinforce.max_reward])

        # init_seeds = rewards
        lx1 = np.append(lx1, [num*batch_size + n for n in range(batch_size)])
        lx2 = np.append(lx2, [num*batch_size + n for n in range(batch_size)])

        line1.set_data(np.array([lx1, ly1]))
        line2.set_data(np.array([lx2, ly2]))
        return [line1, line2]

    fig1 = plt.figure()

    l1, = plt.semilogx([], [], 'r-')
    l2, = plt.semilogx([], [], 'b-')
    plt.xlabel('steps')
    # plt.title('p2, diff softmax, normal dist., rand seed, not dec')
    plt.title('ackley, 100 values, 10 dims, random')
    # plt.title('dixonprice, 100 values, 10 dims, stochastic')
    nb_iter = 1000
    plt.xlim(1, nb_iter)
    plt.ylim(-40, 0)
    line_ani = animation.FuncAnimation(fig1, update_line, nb_iter, fargs=(l1, l2, minimas), interval=10, blit=True, repeat=False)
    plt.show()


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
    # func, I, minimas = ackley_()
    # func, I, minimas = dixonprice_()
    func, I, minimas = polynome_2()
    test_fixed_num_layers(func, I, minimas)

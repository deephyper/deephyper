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


def test_fixed_num_layers(func):
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    #state_space.add_state('x1', [x for x in range(10)])
    state_space.add_state('x1', [4,3,2,1])
    state_space.add_state('x2', [4, 3, 2, 1])
    state_space.add_state('x3', [4,3,2,1])

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = 4

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                1, #async
                                global_step,
                                state_space=state_space)

    #init_seeds = [1. * i / batch_size for i in range(batch_size)]
    get_seeds = lambda x : [float(np.random.uniform(-1,1))*10] * x

    max_reward = [0]
    map = {}

    init_seeds = [0.5 for x in range(batch_size)]

    def update_line(num, max_reward, line1, line2):
        global init_seeds, prev_rewards
        init_seeds = [0.5 for x in range(batch_size)]

        actions = []
        for b in range(batch_size):
            action = reinforce.get_actions([init_seeds[b]], max_layers)
            actions.append(action)
        random.shuffle(actions)
        actions = np.stack(actions).reshape((-1,))
        rewards = []
        prev_rewards = rewards
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            reward = func(conv_action)
            rewards.append(reward)
            map[reward] = init_seeds
            max_reward[0] = max(reward, max_reward[0])
            print(f'STEP = {num} actions: {action} exp: {reinforce.exploration} rewards: {max(rewards)} ema: {reinforce.rewards_b} max_rewards: {max_reward}')
            #if prev_rewards == rewards:
            #     init_seeds = [random.random() for x in range(batch_size)]
            #prev_rewards = rewards

            reinforce.storeRollout(action, [reward], max_layers)
            reinforce.train_step(max_layers, [init_seeds[n]])

        lx1, ly1 = line1.get_data()
        lx2, ly2 = line2.get_data()
        lx1 = np.append(lx1, [num])
        lx2 = np.append(lx2, [num])
        ly1 = np.append(ly1, [reward])
        ly2 = np.append(ly2, [max_reward[0]])
        line1.set_data(np.array([lx1, ly1]))
        line2.set_data(np.array([lx2, ly2]))
        return [line1, line2]

    fig1 = plt.figure()

    l1, = plt.plot([], [], 'r-')
    l2, = plt.plot([], [], 'b-')
    #plt.ylim(0, 20)
    plt.xlabel('steps')
    plt.title('test')
    nb_iter = 1000
    plt.xlim(0, nb_iter)
    plt.ylim(0, 10000)
    line_ani = animation.FuncAnimation(fig1, update_line, nb_iter, fargs=(max_reward, l1, l2), interval=50, blit=True, repeat=False)
    plt.show()


def test_scheduled_num_layers(func):
    pass

def add(v):
    return -sum(v)

def powell_(v):
    return -powell(v)

if __name__ == '__main__':
    test_fixed_num_layers(powell)

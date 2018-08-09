import os
import sys
import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(1000003)
np.random.seed(1000003)

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
    state_space.add_state('x1', [1, 2, 3, 4])
    state_space.add_state('x2', [1, 2, 3, 4])
    state_space.add_state('x3', [1, 2, 3, 4])

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = 1

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                batch_size,
                                global_step,
                                state_space=state_space)

    #init_seeds = [1. * i / batch_size for i in range(batch_size)]
    get_seeds = lambda x : [float(np.random.uniform(-1,1))*10] * x

    max_reward = 0
    map = {}
    for i in range(1, 10000):

        #if i < 100:
        init_seeds = [random.random() for x in range(batch_size)]
        #else:
        #    init_seeds = map[max_reward]
        actions = reinforce.get_actions(init_seeds, max_layers)
        rewards = []
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            #print(f'action: {action} conv_action: {conv_action}')
            reward = func(conv_action)
            rewards.append(reward)
            map[reward] = init_seeds
            max_reward = max(reward, max_reward)
        print(f'STEP = {i} actions: {actions} exp: {reinforce.exploration} rewards: {max(rewards)} max_reward: {max_reward}')
        #print(f'actions: {actions}')
        #print(f'rewards: {rewards}')
        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, init_seeds)

def test_scheduled_num_layers(func):
    pass

def add(v):
    return sum(v)

if __name__ == '__main__':
    test_fixed_num_layers(add)

import os
import sys
import tensorflow as tf
import numpy as np
tf.set_random_seed(1000003)
np.random.seed(1000003)

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV5
from deephyper.search.nas.reinforce.tf import BasicReinforceV5
from deephyper.model.arch import StateSpace


def test_fixed_num_layers(func):
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    state_space.add_state('x1', [1, 2, 3, 4])
    state_space.add_state('x2', [1, 2, 3, 4])
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

    init_seeds = [1. * i / batch_size for i in range(batch_size)]

    for i in range(1, 100):
        actions = reinforce.get_actions(init_seeds, max_layers)
        rewards = []
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            rewards.append(func(action))
        print(f'\nSTEP = {i}')
        print(f'actions: {actions}')
        print(f'rewards: {rewards}')
        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, init_seeds)

def test_scheduled_num_layers(func):
    pass

def add(v):
    return sum(v)

if __name__ == '__main__':
    test_fixed_num_layers(add)

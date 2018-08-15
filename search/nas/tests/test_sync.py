import os
import sys
import tensorflow as tf
import numpy as np
import random
import math
tf.set_random_seed(1000003)
np.random.seed(1000003)
random.seed(1000003)

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


def get_seeds(x):
    return [random.random() for _ in range(x)]

def mean(x):
    return sum(x)/len(x)

def test_fixed_num_layers(func, exploration = 0.5, expl_decay =0.1, fig_name = 'test'):
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    #state_space.add_state('x1', [x for x in range(10)])
    values = [i for i in range(10)]
    values.reverse()
    state_space.add_state('x1', values)
    state_space.add_state('x2', values)
    state_space.add_state('x3', values)
    state_space.add_state('x4', values)
    state_space.add_state('x5', values)
    state_space.add_state('x6', values)
    state_space.add_state('x7', values)
    state_space.add_state('x8', values)
    state_space.add_state('x9', values)
    state_space.add_state('x10', values)

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
                                batch_size,
                                global_step,
                                state_space=state_space,
                                 exploration = exploration,
                                 exploration_decay = expl_decay)

    #init_seeds = [1. * i / batch_size for i in range(batch_size)]

    max_reward = [0]
    map = {}

    init_seeds = [0.5 for x in range(batch_size)]

    reinforce.past_actions = []

    def update_line(num, max_reward, line1, line2):
        global init_seeds, prev_rewards, past_actions
        #init_seeds = [0.5 for x in range(batch_size)]
        init_seeds = get_seeds(batch_size)
        reinforce.past_actions = reinforce.past_actions[-batch_size*2:]
        past_actions = reinforce.past_actions
        #print(f'past actions : {reinforce.past_actions}')
        while True:
            actions = reinforce.get_actions(init_seeds, max_layers)
            duplicate = False
            for n in range(batch_size):
                action = actions[n:len(actions):batch_size]
                if '_'.join([str(x) for x in action]) in past_actions: duplicate = True
            if not duplicate: break
            #else: print('duplicate action')
        rewards = []
        prev_rewards = rewards
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            reinforce.past_actions.append('_'.join([str(x) for x in action]))
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            reward = func(conv_action)
            rewards.append(reward)
            map[reward] = init_seeds
        try:
            print(f'STEP = {num} actions: {action} exp: {reinforce.exploration} rewards: {rewards} max_rewards: {reinforce.max_reward} ema: {reinforce.rewards_b} expl: {reinforce.curr_step_exp}')
        except:
            pass
        # if prev_rewards == rewards:
        #     init_seeds = [random.random() for x in range(batch_size)]
        #prev_rewards = rewards

        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, init_seeds)

        lx1, ly1 = line1.get_data()
        lx2, ly2 = line2.get_data()
        #lx3, ly3 = line3.get_data()
        lx1 = np.append(lx1, [num])
        lx2 = np.append(lx2, [num])
        #lx3 = np.append(lx3, [num])
        ly1 = np.append(ly1, max(rewards))
        ly2 = np.append(ly2, [reinforce.max_reward])
        #ly3 = np.append(ly3, [reinforce.rewards_b])
        line1.set_data(np.array([lx1, ly1]))
        line2.set_data(np.array([lx2, ly2]))
        #line3.set_data(np.array([lx3, ly3]))
        return [line1, line2]

    fig1 = plt.figure()

    l1, = plt.plot([], [], 'r-')
    l2, = plt.plot([], [], 'b-')
    l3, = plt.plot([],[], 'g-')
    #plt.ylim(0, 20)
    plt.yscale('log')
    plt.xlabel('steps')
    plt.title('test')
    nb_iter = 10
    plt.xlim(0, nb_iter)
    plt.ylim(100000000, 1)

    line_ani = animation.FuncAnimation(fig1, update_line, nb_iter, fargs=(max_reward, l1, l2), interval=10, blit=True, repeat=False)
    #plt.show()
    plt.savefig('figures/'+fig_name+'.pdf')
    print('saved')


def test_fixed_num_layers_v2(func, exploration = 0.5, expl_decay =0.1, fig_name = 'test', nb_iter=500):
    tf.reset_default_graph()
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    #state_space.add_state('x1', [x for x in range(10)])
    values = [i for i in range(10)]
    values.reverse()
    state_space.add_state('x1', values)
    state_space.add_state('x2', values)
    state_space.add_state('x3', values)
    state_space.add_state('x4', values)
    state_space.add_state('x5', values)
    state_space.add_state('x6', values)
    #state_space.add_state('x7', values)
    #state_space.add_state('x8', values)
    #state_space.add_state('x9', values)
    #state_space.add_state('x10', values)

    policy_network = NASCellPolicyV5(state_space)
    max_layers = 1
    batch_size = 1

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                batch_size,
                                global_step,
                                state_space=state_space,
                                 exploration = exploration,
                                 exploration_decay = expl_decay)

    #init_seeds = [1. * i / batch_size for i in range(batch_size)]

    max_reward = [0]
    map = {}

    init_seeds = [0.5 for x in range(batch_size)]

    reinforce.past_actions = []

    rewards_plt = []
    max_plt = []

    for iter in range(nb_iter):
        #init_seeds = [0.5 for x in range(batch_size)]
        init_seeds = get_seeds(batch_size)
        reinforce.past_actions = reinforce.past_actions[-batch_size*2:]
        past_actions = reinforce.past_actions
        #print(f'past actions : {reinforce.past_actions}')
        num_attempts = 10
        while True:
            init_seeds = get_seeds(batch_size)
            actions = reinforce.get_actions(init_seeds, max_layers)
            duplicate = False
            num_attempts -=1
            for n in range(batch_size):
                action = actions[n:len(actions):batch_size]
                if '_'.join([str(x) for x in action]) in past_actions: duplicate = True
            if not duplicate: break
            elif not num_attempts:
                reinforce.exploration *= math.exp(reinforce.exploration_decay)
                break
            #else: print('duplicate action')
        rewards = []
        prev_rewards = rewards
        for n in range(batch_size):
            action = actions[n:len(actions):batch_size]
            reinforce.past_actions.append('_'.join([str(x) for x in action]))
            conv_action = state_space.parse_state(action, num_layers=max_layers)
            reward = -1.*func(conv_action)
            rewards.append(reward)
            map[reward] = init_seeds
        rewards_plt.append(max(rewards))
        max_plt.append(reinforce.max_reward)
        #try:
        print(f'STEP = {iter} actions: {action} exp: {reinforce.exploration} rewards: {rewards} max_rewards: {reinforce.max_reward} ema: {reinforce.rewards_b} expl: {reinforce.curr_step_exp}')
        #except:
        #    pass
        # if prev_rewards == rewards:
        #     init_seeds = [random.random() for x in range(batch_size)]
        #prev_rewards = rewards

        reinforce.storeRollout(actions, rewards, max_layers)
        reinforce.train_step(max_layers, init_seeds)

        # lx1, ly1 = line1.get_data()
        # lx2, ly2 = line2.get_data()
        # #lx3, ly3 = line3.get_data()
        # lx1 = np.append(lx1, [num])
        # lx2 = np.append(lx2, [num])
        # #lx3 = np.append(lx3, [num])
        # ly1 = np.append(ly1, max(rewards))
        # ly2 = np.append(ly2, [reinforce.max_reward])
        # #ly3 = np.append(ly3, [reinforce.rewards_b])
        # line1.set_data(np.array([lx1, ly1]))
        # line2.set_data(np.array([lx2, ly2]))
        # #line3.set_data(np.array([lx3, ly3]))
        # return [line1, line2]

    return rewards_plt, max_plt

def test_scheduled_num_layers(func):
    pass

def add(v):
    return sum(v)

def powell_(v):
    return -powersum(v)

if __name__ == '__main__':
    allfuncs = [
        add,
        polynome_2,
        ackley,
        dixonprice,
        ellipse,
        griewank,
        levy,
        michalewicz,  # min < 0
        nesterov,
        perm,
        powell,
        # powellsincos,  # many local mins
        powersum,
        rastrigin,
        roscenbrock,
        schwefel,  # many local mins
        sphere,
        saddle,
        sum2,
        trid,  # min < 0
        zakharov
    ]
    allfuncs_str = [
        'add',
        'polynome_2'
        'ackley',
        'dixonprice',
        'ellipse',
        'griewank',
        'levy',
        'michalewicz',  # min < 0
        'nesterov',
        'perm',
        'powell',
        # powellsincos,  # many local mins
        'powersum',
        'rastrigin',
        'roscenbrock',
        'schwefel',  # many local mins
        'sphere',
        'saddle',
        'sum2',
        'trid',  # min < 0
        'zakharov'
    ]

    for fun_i in range(len(allfuncs)):
        exp = 0.0
        fig_name = f'exp={exp}'
        all_rewards = []
        #try:
        ax1 = plt.subplot(131)
        ax1.margins(1)
        print(f'\n\n {fig_name}')
        rewards_plt, max_plt = test_fixed_num_layers_v2(allfuncs[fun_i], exploration=exp, expl_decay=0.0, fig_name=fig_name, nb_iter=100)
        ax1.plot(range(len(rewards_plt)), rewards_plt, 'b')
        ax1.plot(range(len(rewards_plt)), max_plt, 'k')
        ax1.set_title(fig_name)
        ax1.set_ylim(np.percentile(rewards_plt,5),0)
        ax1.set_xlim(0,100)
        all_rewards.extend(rewards_plt)
        #except: pass
        exp = 0.5
        fig_name = f'exp={exp}'
        #try:
        ax2 = plt.subplot(132)
        ax2.margins(1)
        print(f'\n\n {fig_name}')
        rewards_plt, max_plt = test_fixed_num_layers_v2(allfuncs[fun_i], exploration=exp, expl_decay=0.1, fig_name=fig_name, nb_iter=100)
        ax2.plot(range(len(rewards_plt)), rewards_plt, 'b')
        ax2.plot(range(len(rewards_plt)), max_plt, 'k')
        ax2.set_title(fig_name)
        ax2.set_ylim(np.percentile(rewards_plt, 5), 0)
        ax2.set_xlim(0, 100)
        all_rewards.extend(rewards_plt)

        #except: pass
        exp = 1.0
        fig_name = f'exp={exp}[RANDOM]'
        #try:
        ax3 = plt.subplot(133)
        ax3.margins(1)
        print(f'\n\n {fig_name}')
        rewards_plt, max_plt = test_fixed_num_layers_v2(allfuncs[fun_i], exploration=exp, expl_decay=0, fig_name=fig_name, nb_iter=100)
        ax3.plot(range(len(rewards_plt)), rewards_plt, 'b')
        ax3.plot(range(len(rewards_plt)), max_plt, 'k')
        ax3.set_title(fig_name)
        ax3.set_ylim(np.percentile(rewards_plt, 5), 0)
        ax3.set_xlim(0, 100)
        all_rewards.extend(rewards_plt)
        #except: pass


        # l1, = plt.plot(range(len(rewards_plt)), rewards_plt, 'r-')
        # l2, = plt.plot(range(len(rewards_plt)), max_plt, 'b-')
        # plt.yscale('log')
        # plt.xlabel('steps')
        # plt.title(fig_name)
        #plt.xlim(0, 0)
        #plt.ylim(np.percentile(all_rewards,5),10)
        fig_name = f'func: -{allfuncs_str[fun_i]}'
        fig = plt.gcf()
        fig.set_size_inches(18, 8)
        plt.savefig('figures_all/'+fig_name+'.pdf')
        plt.close()
        print('saved')

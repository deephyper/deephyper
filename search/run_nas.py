import datetime
import glob
import os
import pickle
import signal
import sys
from collections import OrderedDict
from math import ceil, log
from pprint import pprint
from random import random
from time import ctime, time, sleep
from importlib import import_module, reload

import numpy as np
import tensorflow as tf
tf.set_random_seed(1000003)
np.random.seed(1000003)

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

import deephyper.model.arch as a
from deephyper.evaluators import evaluate
from deephyper.search import util
from deephyper.search.nas.policy.tf import NASCellPolicyV5
from deephyper.search.nas.reinforce.tf import BasicReinforceV5

logger = util.conf_logger('deephyper.search.run_nas')

import subprocess as sp
logger.debug(f'ddd {sp.Popen("which mpirun".split())}')
logger.debug(f'python exe : {sys.executable}')

from balsam.launcher import dag
from balsam.launcher import worker

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations

class Search:
    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator_nas(cfg)
        self.config = cfg.config
        self.map_model_reward = {}

    def get_reward(self, model):
        '''
        Args
            model: list of index corresponding to tokens in StateSpace
        Return
            the corresponding reward if the model is already None, if the model is not
            known returns None
        '''
        hash_key = str(model)
        reward = self.map_model_reward.get(hash_key)
        return reward

    def set_reward(self, model, reward):
        '''
        Args
            model: list of index corresponding to tokens in StateSpace
            reward: corresponding reward of model
        '''
        hash_key = str(model)
        self.map_model_reward[hash_key] = reward

    def run(self):
        print(self.config)
        model_path = self.config[a.model_path]
        policy_network = NASCellPolicyV5(self.config[a.state_space], save_path=model_path)
        self.max_layers = self.config[a.max_layers]
        start_layers = self.config[a.min_layers] if a.min_layers in self.config else 2
        assert self.max_layers > 0

        self.global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(0.99,
                                                   self.global_step,
                                                   500,
                                                   0.96,
                                                   staircase=True)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        if (self.opt_config.sync):
            logger.debug('Run synchronous...')
            self.run_sync(policy_network, optimizer, learning_rate, start_layers)
        else:
            logger.debug('Run asynchronous...')
            self.run_async(policy_network, optimizer, learning_rate, start_layers)

    def run_async(self, policy_network, optimizer, learning_rate, num_layers):
        '''
            Asynchronous approach.
        '''
        session = tf.Session()
        num_workers = self.opt_config.num_workers
        state_space = self.config[a.state_space]
        children_exp = 0 # number of childs networks trainned since the last best reward
        best_reward = 0.
        # for the CONTROLLER
        controller_batch_size = 1 # 1 for asynchronous NUM_WORKERS for synchronous
        reinforce = BasicReinforceV5(session,
                                   optimizer,
                                   policy_network,
                                   self.max_layers,
                                   1, #asynchronous
                                   self.global_step,
                                   state_space=state_space)

        # Init State
        logger.debug(f'num_workers = {num_workers}')
        step = 0
        worker_steps = [ 0 for i in range(num_workers)]

        #for n, state in enumerate(states):
        for n in range(num_workers):
            init_seed = [float(np.random.uniform(-1,1))]*controller_batch_size
            action = reinforce.get_actions(rnn_input=init_seed,
                                           num_layers=num_layers)
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['num_worker'] = n
            cfg['num_layers'] = num_layers
            cfg['step'] = 0
            cfg['init_seed'] = init_seed
            cfg['arch_seq'] = action
            self.evaluator.add_eval_nas(cfg)

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)

        controller_patience= 5 * num_workers
        results = []
        for elapsed_str in timer:
            new_results = list(self.evaluator.get_finished_evals())
            results.extend(new_results)
            len_results = len(results)
            logger.debug("[ Time = {0}, Step = {1} : results = {2} ]".format(elapsed_str, step, len_results))
            children_exp += len_results

            # Get rewards and apply reinforcement step by step
            for cfg, reward in results:
                if (reward > best_reward):
                    best_reward = reward
                    children_exp = 0
                state = cfg['arch_seq']
                logger.debug(f'--> seed: {cfg["init_seed"]} , reward: {reward} , arch_seq: {cfg["arch_seq"]} , num_layers: {cfg["num_layers"]}')
                reinforce.storeRollout(state, [reward], num_layers)
                step += 1
                reinforce.train_step(num_layers, cfg['init_seed'])

            # Check improvement of children NN
            if (children_exp > controller_patience): # add a new layer to the search
                if (num_layers < self.max_layers):
                    num_layers += 1
                    children_exp = 0
                else:
                    logger.debug('Best accuracy is not increasing')

            # Run training on new children NN
            next_results = []
            for cfg, reward in results:
                #state = cfg['arch_seq']
                num_worker = cfg['num_worker']
                init_seed = [float(np.random.uniform(-1, 1))] * controller_batch_size

                action = reinforce.get_actions(rnn_input=init_seed,
                                               num_layers=num_layers)
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['init_seed'] = init_seed
                cfg['arch_seq'] = action
                cfg['num_layers'] = num_layers
                cfg['num_worker'] = num_worker
                worker_steps[num_worker] += 1
                cfg['step'] = worker_steps[num_worker]
                supposed_reward = self.get_reward(cfg['arch_seq'])
                if supposed_reward == None:
                    self.evaluator.add_eval_nas(cfg)
                    logger.debug('add_evals_nas')
                else:
                    next_results.append((cfg, supposed_reward))
                logger.debug(f' steps = {worker_steps}')
            results = next_results

    def run_sync(self, policy_network, optimizer, learning_rate, num_layers):
        '''
            Batch Synchronous algorithm for NAS.
        '''
        session = tf.Session()
        num_workers = self.opt_config.num_workers
        state_space = self.config[a.state_space]
        children_exp = 0 # number of childs networks trainned since the last best reward
        best_reward = 0.
        # for the CONTROLLER
        controller_batch_size = num_workers # 1 for asynchronous NUM_WORKERS for synchronous
        reinforce = BasicReinforceV5(session,
                                   optimizer,
                                   policy_network,
                                   self.max_layers,
                                   controller_batch_size, #asynchronous
                                   self.global_step,
                                   state_space=state_space)
        num_tokens = state_space.get_num_tokens(num_layers)
        # Init State
        step = 0
        worker_steps = [ 0 for i in range(num_workers)]

        cfg_list = []
        init_seeds = [float(np.random.uniform(-1,1))]*controller_batch_size
        actions = reinforce.get_actions(rnn_input=init_seeds,
                                        num_layers=num_layers)
        for n in range(num_workers):
            action = actions[n:len(actions):num_workers]
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['num_worker'] = n
            cfg['num_layers'] = num_layers
            cfg['step'] = 0
            cfg['init_seed'] = init_seeds[n]
            cfg['arch_seq'] = action
            self.evaluator.add_eval_nas(cfg)
            cfg_list.append(cfg)

        controller_patience = 5 * num_workers

        while True:
            results = self.evaluator.await_evals(cfg_list)
            logger.debug("results received")

            # Get rewards and apply reinforcement step by step
            states = []
            rewards = []
            init_seeds = []
            results_list = []
            for cfg, reward in results:
                results_list.append((cfg,reward))
                children_exp += 1
                if (reward > best_reward):
                    best_reward = reward
                    children_exp = 0
                states.append(cfg['arch_seq'])
                rewards.append(reward)
                init_seeds.append(cfg['init_seed'])
                logger.debug(f'--> seed: {cfg["init_seed"]} , reward: {reward} , arch_seq: {cfg["arch_seq"]} , num_layers: {cfg["num_layers"]}')

            states = join_states(states)
            reinforce.storeRollout(states, rewards, num_layers)
            step += 1
            reinforce.train_step(num_layers, init_seeds)

            # Check improvement of children NN
            if (children_exp > controller_patience): # add a new layer to the search
                if (num_layers < self.max_layers):
                    num_layers += 1
                    children_exp = 0
                else:
                    logger.debug('Best accuracy is not increasing')

            actions = reinforce.get_actions(rnn_input=init_seeds,
                                            num_layers=num_layers)

            # Run training on new children NN
            cfg_list = []
            for n in range(num_workers):
                action = actions[n:len(actions):num_workers]
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['num_worker'] = n
                cfg['num_layers'] = num_layers
                cfg['init_seed'] = init_seeds[n]
                cfg['arch_seq'] = action
                worker_steps[n] += 1
                cfg['step'] = worker_steps[n]
                self.evaluator.add_eval_nas(cfg)
                cfg_list.append(cfg)
                logger.debug('add_evals_nas')


def join_states(states):
    states = np.array(states)
    res = []
    for t in range(len(states[0])):
        res = res + states[:,t].tolist()
    return res


def main(args):
    '''Service loop: add jobs; read results; drive nas'''

    cfg = util.OptConfigNas(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS on benchmark {cfg.benchmark} & run with {cfg.run_module_name}")
    controller.run()

def test_join_states():
    l1 = [3., 1., 1., 1., 0., 0., 4., 0., 0., 0., 3., 3., 3., 3., 1.]
    l2 = [3., 1., 1., 1., 0., 0., 4., 0., 0., 0., 3., 3., 3., 3., 1.]
    l3 = [3., 1., 1., 1., 0., 0., 4., 0., 0., 0., 3., 3., 3., 3., 1.]
    l = [l1, l2, l3]
    print(join_states(l))

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

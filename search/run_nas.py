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

class Search_old:
    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator_nas(cfg)
        self.config = cfg.config

    def run(self):
        NUM_WORKERS = self.opt_config.num_workers
        session = tf.Session()
        global_step = tf.Variable(0, trainable=False)
        state_space = self.config[a.state_space]
        policy_network = NASCellPolicyV5(state_space)
        max_layers = self.config[a.max_layers]
        assert max_layers > 0
        num_of_layers = max_layers if max_layers >= 2 else 2
        children_exp = 0 # number of childs networks trainned since the last best reward
        best_reward = 0.
        last_reward = 0.

        learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                   500, 0.96, staircase=True)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # for the CONTROLLER
        reinforce = BasicReinforceV5(session,
                                   optimizer,
                                   policy_network,
                                   max_layers,
                                   global_step,
                                   num_features=state_space.size,
                                   state_space=state_space)

        # Init State
        logger.debug(f'num_workers = {NUM_WORKERS}')
        states = np.array(state_space.get_random_state_space(num_of_layers,
                                                             num=NUM_WORKERS),
                          dtype=np.float32)
        step = 0
        steps = [ 0 for i in range(len(states))]

        for n, state in enumerate(states):
            action = reinforce.get_action(state=np.array([state], dtype=np.float32),                                    num_layers=num_of_layers)
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['num_worker'] = n
            cfg['step'] = 0
            cfg['arch_seq'] = action.tolist()
            self.evaluator.add_eval_nas(cfg)

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        for elapsed_str in timer:
            results = list(self.evaluator.get_finished_evals())
            len_results = len(results)
            logger.debug("[ Time = {0}, Step = {1} : results = {2} ]".format(elapsed_str, step, len_results))
            children_exp += len_results

            # Get rewards and apply reinforcement step by step
            for cfg, reward in results:
                if (reward > best_reward):
                    best_reward = reward
                    children_exp = 0
                state = cfg['arch_seq']
                logger.debug(f'state = {state}')
                reinforce.storeRollout(state, reward)
                step += 1
                ls = reinforce.train_step(1)

            # Check improvement of children NN
            if (children_exp > 100): # add a new layer to the search
                if (num_of_layers < max_layers):
                    num_of_layers += 1
                    for cfg, _ in results:
                        state_space.extends_num_layer_of_state(cfg['arch_seq'], num_of_layers)
                else:
                    logger.debug('Best accuracy is not increasing')

            # Run training on new children NN
            for cfg, reward in results:
                state = cfg['arch_seq']
                num_worker = cfg['num_worker']
                action = reinforce.get_action(state=np.array(state, dtype=np.float32),
                                              num_layers=num_of_layers)
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['arch_seq'] = action.tolist()
                cfg['num_worker'] = num_worker
                steps[num_worker] += 1
                cfg['step'] = steps[num_worker]
                self.evaluator.add_eval_nas(cfg)
                logger.debug('add_evals_nas')
                logger.debug(f' steps = {steps}')

class Search:
    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator_nas(cfg)
        self.config = cfg.config

    def run(self):
        NUM_WORKERS = self.opt_config.num_workers
        session = tf.Session()
        global_step = tf.Variable(0, trainable=False)
        state_space = self.config[a.state_space]
        print(self.config)
        model_path = self.config[a.model_path]
        policy_network = NASCellPolicyV5(state_space,save_path=model_path)
        max_layers = self.config[a.max_layers]
        start_layers = self.config[a.min_layers] if a.min_layers in self.config else 2
        assert max_layers > 0
        num_of_layers = start_layers if start_layers >= 2 else 2
        children_exp = 0 # number of childs networks trainned since the last best reward
        best_reward = 0.
        last_reward = 0.

        learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # for the CONTROLLER
        controller_batch_size = 1 # 1 for asynchronous NUM_WORKERS for synchronous
        reinforce = BasicReinforceV5(session,
                                   optimizer,
                                   policy_network,
                                   max_layers,
                                   1, #asynchronous
                                   global_step,
                                   num_features=state_space.size,
                                   state_space=state_space)

        # Init State
        logger.debug(f'num_workers = {NUM_WORKERS}')
        #states = np.array(state_space.get_random_state_space(num_of_layers,
        #                                                     num=NUM_WORKERS),
        #                  dtype=np.float32)
        step = 0
        worker_steps = [ 0 for i in range(NUM_WORKERS)]

        #for n, state in enumerate(states):
        for n in range(NUM_WORKERS):
            action = reinforce.get_actions(rnn_input=[float(np.random.uniform(-1,1))]*controller_batch_size,                                    num_layers=num_of_layers)
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['num_worker'] = n
            cfg['num_layers'] = num_of_layers
            cfg['step'] = 0
            cfg['arch_seq'] = action
            self.evaluator.add_eval_nas(cfg)

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)

        controller_patience=5*NUM_WORKERS

        for elapsed_str in timer:
            results = list(self.evaluator.get_finished_evals())
            len_results = len(results)
            logger.debug("[ Time = {0}, Step = {1} : results = {2} ]".format(elapsed_str, step, len_results))
            children_exp += len_results

            # Get rewards and apply reinforcement step by step
            for cfg, reward in results:
                if (reward > best_reward):
                    best_reward = reward
                    children_exp = 0
                state = cfg['arch_seq']
                logger.debug(f'state = {state}')
                reinforce.storeRollout(state, [reward], num_of_layers)
                step += 1
                ls = reinforce.train_step(num_of_layers, [float(np.random.uniform(-1,1))])

            # Check improvement of children NN
            if (children_exp > controller_patience): # add a new layer to the search
                if (num_of_layers < max_layers):
                    num_of_layers += 1
                    for cfg, _ in results:
                        state_space.extends_num_layer_of_state(cfg['arch_seq'], num_of_layers)
                else:
                    logger.debug('Best accuracy is not increasing')

            # Run training on new children NN
            for cfg, reward in results:
                #state = cfg['arch_seq']
                num_worker = cfg['num_worker']
                action = reinforce.get_actions(rnn_input=[float(np.random.uniform(-1,1))],
                                              num_layers=num_of_layers)
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['arch_seq'] = action
                cfg['num_layers'] = num_of_layers
                cfg['num_worker'] = num_worker
                worker_steps[num_worker] += 1
                cfg['step'] = worker_steps[num_worker]
                self.evaluator.add_eval_nas(cfg)
                logger.debug('add_evals_nas')
                logger.debug(f' steps = {worker_steps}')


def main(args):
    '''Service loop: add jobs; read results; drive nas'''

    cfg = util.OptConfigNas(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS on benchmark {cfg.benchmark} & run with {cfg.run_module_name}")
    controller.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

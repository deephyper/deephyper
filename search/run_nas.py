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
#from balsam.launcher import dag, worker

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

import deephyper.model.arch as a
from deephyper.evaluators import evaluate
from deephyper.search import util
from deephyper.search.nas.policy.tf import NASCellPolicyV2
from deephyper.search.nas.reinforce.tf import BasicReinforce

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
logger = util.conf_logger('deephyper.search.run_nas')

class Search:
    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator_nas(cfg)
        self.config = cfg.config

    def run(self):
        session = tf.Session()
        global_step = tf.Variable(0, trainable=False)
        state_space = self.config[a.state_space]
        policy_network = NASCellPolicyV2(state_space)
        max_layers = self.config[a.max_layers]

        learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                   500, 0.96, staircase=True)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # for the CONTROLLER
        reinforce = BasicReinforce(session,
                                   optimizer,
                                   policy_network,
                                   max_layers,
                                   global_step,
                                   num_features=state_space.size,
                                   state_space=state_space)

        MAX_EPISODES = self.config[a.max_episodes]
        step = 0
        # Init State
        logger.debug(f'num_workers = {self.opt_config.num_workers}')
        states = np.array(self.opt_config.starting_point, dtype=np.float32)

        for state in states:
            action = reinforce.get_action(state=np.array([state], dtype=np.float32))
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['arch_seq'] = action.tolist()
            self.evaluator.add_eval_nas(cfg)

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        nb_iter = 0
        for elapsed_str in timer:
            results = list(self.evaluator.get_finished_evals())
            logger.debug("[ Time = {0}, Step = {1} : results = {2} ]".format(elapsed_str, step, len(results)))
            for cfg, reward in results:
                state = cfg['arch_seq']
                logger.debug(f'state = {state}')
                reinforce.storeRollout(state, reward)
                step += 1
                ls = reinforce.train_step(1)
            for cfg, reward in results:
                state = cfg['arch_seq']
                action = reinforce.get_action(state=np.array(state, dtype=np.float32))
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['arch_seq'] = action.tolist()
                self.evaluator.add_eval_nas(cfg)
                logger.debug('add_evals_nas')
            if nb_iter >= MAX_EPISODES:
                break
            else:
                nb_iter += 1


def main(args):
    '''Service loop: add jobs; read results; drive nas'''

    #cfg = util.OptConfigNas(args, num_workers=len(worker)-2)
    cfg = util.OptConfigNas(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS on benchmark {cfg.benchmark} & run with {cfg.run_module_name}")
    controller.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

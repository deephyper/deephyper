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
from deephyper.search.nas.policy.tf import NASCellPolicyV2
from deephyper.search.nas.reinforce.tf import BasicReinforce

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

    def run(self):
        NUM_WORKERS = self.opt_config.num_workers
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

        # Init State
        logger.debug(f'num_workers = {NUM_WORKERS}')
        states = np.array(self.opt_config.starting_point, dtype=np.float32)
        step = 0
        steps = [ 0 for i in range(len(states))]

        for n, state in enumerate(states):
            action = reinforce.get_action(state=np.array([state], dtype=np.float32))
            cfg = self.config.copy()
            cfg['global_step'] = step
            cfg['num_worker'] = n
            cfg['step'] = 0
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
                num_worker = cfg['num_worker']
                action = reinforce.get_action(state=np.array(state, dtype=np.float32))
                cfg = self.config.copy()
                cfg['global_step'] = step
                cfg['arch_seq'] = action.tolist()
                cfg['num_worker'] = num_worker
                steps[num_worker] += 1
                cfg['step'] = steps[num_worker]
                self.evaluator.add_eval_nas(cfg)
                logger.debug('add_evals_nas')
                logger.debug(f' steps = {steps}')

def main(args):
    '''Service loop: add jobs; read results; drive nas'''

    #cfg = util.OptConfigNas(args, num_workers=len(list(worker.WorkerGroup()))-2)
    #logger.debug(f'wokers = {list(worker.WorkerGroup())}')
    cfg = util.OptConfigNas(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS on benchmark {cfg.benchmark} & run with {cfg.run_module_name}")
    controller.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

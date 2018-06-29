import tensorflow as tf
import numpy as np
import glob

from random import random
from math import log, ceil
from time import time, ctime
import datetime

import logging
import os
import pickle
import signal
import sys
from pprint import pprint

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util
import deephyper.model.arch as a
from deephyper.search.controller.nas.policy.tf import NASCellPolicy
from deephyper.search.controller.nas.reinforce.tf import BasicReinforce
from deephyper.model.trainer.tf import BasicTrainer
from deephyper.model.utilities.conversions import action2dict

logger = util.conf_logger('deephyper.search.nas')

class Search:
    def __init__(self, cfg):
        self.opt_config = cfg
        self.evaluator = evaluate.create_evaluator(cfg)
        self.config = cfg.space_dict

    def run(self):
        session = tf.Session()
        global_step = tf.Variable(0, trainable=False)
        num_features = len(self.config[a.features])
        policy_network = NASCellPolicy(num_features=num_features)
        max_layers = self.config[a.max_layers]

        learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                   500, 0.96, staircase=True)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # for the CONTROLLER
        reinforce = BasicReinforce(session, optimizer, policy_network, max_layers, global_step,
            num_features)

        MAX_EPISODES = self.config[a.max_episodes]
        step = 0
        # Init State
        state = np.array(
            [[1.0 for i in range(len(self.config[a.features]))]*max_layers], dtype=np.float32)
        total_rewards = 0
        for i_episode in range(MAX_EPISODES):
            action = reinforce.get_action(state=state)
            architecture = action2dict(self.config, action[0][0])
            print("[ Episode = {0} ] action = {1}".format(i_episode, architecture))
            if all(ai > 0 for ai in action[0][0]):
                # training the generated CNN and get the reward
                self.config['global_step'] = i_episode
                print("HERE")
                pprint(self.config)
                self.evaluator.add_eval(self.config)
                rewards = self.evaluator.await_evals([self.config])
                #print("[ Episode = {0} ] reward = {1}".format(i_episode, res[0]))
            else:
                rewards = [-1.0]
            for reward in rewards:
                total_rewards += reward
                print("R = ", reward)

            # In our sample action is equal state
            state = action[0]
            reinforce.storeRollout(state, reward)

            step += 1
            ls = reinforce.train_step(1)
            log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(
                i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
            log = open("lg3.txt", "a+")
            log.write(log_str)
            log.close()
            print(log_str)

def main(args):
    '''Service loop: add jobs; read results; drive nas'''

    cfg = util.OptConfig(args)
    controller = Search(cfg)
    logger.info(f"Starting new NAS run with {cfg.benchmark_module_name}")
    controller.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

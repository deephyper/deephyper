"""Game class to represent 2048 game state."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from multiprocessing.pool import ThreadPool
from tensorforce.environments import Environment
import numpy as np

from deephyper.evaluators import evaluate


def submit_task(evaluator, cfg):
    evaluator.add_eval_nas(cfg)
    results = evaluator.await_evals([cfg])
    for cfg, reward in results:
        return reward


class AsyncNasBalsamEnvironment(Environment):

    def __init__(self, opt_config, pool, state=np.array([1.]), initial_score=0):

        self.opt_config = opt_config
        self.state_space = opt_config.config['state_space']
        self.evaluator = evaluate.create_evaluator_nas(opt_config)

        self._score = initial_score
        self._state = state
        self.action_buffer = []
        self.num_timesteps = self.state_space.size * self.state_space.num_blockss


    def __str__(self):
        self.print_state()

    def reset(self):
        self.__init__(self.opt_config)
        return self._state

    def execute(self, action):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            terminal = False
            reward = 0
            self._state = [float(action)]
            return self._state, terminal, reward

        conv_action = self.state_space.parse_state(self.action_buffer)
        # print(f'conv_action: {conv_action}')

        terminal = True
        self.action_buffer = []
        self._state = [1.]

        cfg = self.opt_config.config.copy()
        cfg['arch_seq'] = conv_action

       # async_reward = self.pool.apply_async(submit_task, (self.evaluator, cfg,))
        async_reward = self.evaluator.apply_async(cfg)


        return self._state, terminal, async_reward

    @property
    def states(self):
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        num_actions = self.state_space.max_num_classes
        return dict(num_actions=num_actions, type='int')

    def copy(self):
        """Return a copy of self."""

        return AsyncNasBalsamEnvironment(self.opt_config, np.copy(self._state), self._score)

    def state(self):
        """Return current state."""
        return self._state

    def score(self):
        """Return current score."""
        return self._score

    def print_state(self):
        print(self.state)

"""Game class to represent 2048 game state."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from multiprocessing.pool import ThreadPool
from benchmark_functions_wrappers import *
from tensorforce.environments import Environment
import numpy as np


NUM_DIM = 10
ACTION_NAMES = np.linspace(-2, 2, NUM_DIM)
NUM_ACTIONS = len(ACTION_NAMES)

math_func, _, _ = ackley_()
# math_func, _, _ = polynome_2()
# math_func, _, _ = dixonprice_()
# math_func, _, _ = levy_()
# math_func, _, _ = griewank_()

POOL = ThreadPool(processes=10)

possible_times = np.linspace(0, 3, 30)

def math_func_sleep(v):
    elps = np.random.choice(possible_times)
    time.sleep(elps)
    return math_func(v)



class DistMathFun(Environment):

    def __init__(self, state=np.array([1.]), initial_score=0):

        self._score = initial_score
        self._state = state
        self.action_buffer = []
        self.num_timesteps = NUM_DIM
        self.pool = POOL

    def __str__(self):
        self.print_state()

    def reset(self):
        self.__init__()
        return self._state

    def execute(self, action):
        self.action_buffer.append(action)
        if len(self.action_buffer) < self.num_timesteps:
            terminal = False
            reward = 0
            self._state = [float(action)]
            return self._state, terminal, reward

        conv_action = [ACTION_NAMES[a] for a in self.action_buffer]
        # print(f'conv_action: {conv_action}')
        terminal = True
        self.action_buffer = []
        self._state = [1.]
        async_reward = self.pool.apply_async(math_func_sleep, (conv_action,))

        return self._state, terminal, async_reward

    @property
    def states(self):
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=NUM_ACTIONS, type='int')

    def copy(self):
        """Return a copy of self."""

        return DistMathFun(np.copy(self._state), self._score)

    def state(self):
        """Return current state."""
        return self._state

    def score(self):
        """Return current score."""
        return self._score

    def print_state(self):
        print(self.state)

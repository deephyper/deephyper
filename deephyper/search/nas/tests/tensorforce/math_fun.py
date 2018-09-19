"""Game class to represent 2048 game state."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from benchmark_functions_wrappers import *
from tensorforce.environments import Environment
import numpy as np

# ACTION_NAMES = np.linspace(-2, 2, 10)
# NUM_ACTIONS = len(ACTION_NAMES)
# NUM_DIM = 10

# math_func, _, _ = ackley_()
# math_func, _, _ = polynome_2()
# math_func, _, _ = dixonprice_()
# math_func, _, _ = levy_()
# math_func, _, _ = griewank_()

# def math_func_sleep(v):
#     # time.sleep(1)
#     return math_func(v)

class MathFun(Environment):

    def __init__(self, num_dim=10,
                       num_action=10,
                       func_wrapper=ackley_,
                       state=np.array([1.]),
                       initial_score=0):

        self.num_dim = num_dim
        self.num_action = num_action
        self.func_wrapper = func_wrapper
        self._score = initial_score
        self._state = state
        self.init_state = state
        self.action_buffer = []
        self.conv_action = None
        self.math_func, (self.a, self.b), self.optimum = self.func_wrapper()
        self.action_tokens = np.linspace(self.a, self.b, self.num_action)

    def __str__(self):
        self.print_state()

    def reset(self):
        self.__init__(self.num_dim, self.num_action, self.func_wrapper, self.init_state)
        return self._state

    def execute(self, action):
        self.action_buffer.append(action)
        if len(self.action_buffer) < self.num_dim:
            terminal = False
            reward = 0
            self._state = [float(action)]
            return self._state, terminal, reward

        self.conv_action = [self.action_tokens[a] for a in self.action_buffer]
        print(f'action_buffer: {self.action_buffer}')
        # print(f'conv_action: {conv_action}')
        terminal = True
        self.action_buffer = []
        self._state = self.init_state
        reward = self.math_func(self.conv_action)

        return self._state, terminal, reward

    @property
    def states(self):
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=self.num_action, type='int')

    def copy(self):
        """Return a copy of self."""
        return MathFun(self.num_dim, self.num_action, self.func_wrapper, np.copy(self._state), self._score)

    def state(self):
        """Return current state."""
        return self._state

    def score(self):
        """Return current score."""
        return self._score

    def print_state(self):
        print(self.state)

"""Game class to represent 2048 game state."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
from tensorforce.environments import Environment
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) # environment dir
top = os.path.dirname(os.path.dirname(HERE)) # search dir
top = os.path.dirname(os.path.dirname(top)) # folder containing deephyper
sys.path.append(top)

from deephyper.search.nas.environments.distributed_rewards_env import DistributedRewardsEnvironment
from deephyper.search.nas.utils.benchmark_functions_wrappers import *
from deephyper.search.nas.utils.state_space import StateSpace


class NasEnvironment(DistributedRewardsEnvironment):

    def __init__(self, state_space, num_dim=5, init_state=np.array([1.]), initial_score=0):

        assert isinstance(state_space, StateSpace)
        self.state_space = state_space
        self.num_dim = num_dim
        self._state = self.init_state = init_state
        self._score = initial_score
        self.action_buffer = []

    def __str__(self):
        self.print_state()

    def reset(self):
        self.__init__(self.state_space, self.num_dim, self.init_state)
        return self._state

    def execute(self, action):
        self.action_buffer.append(action)
        if len(self.action_buffer) < self.num_dim:
            terminal = False
            reward = 0
            self._state = [float(action)]
            return self._state, terminal, reward

        conv_action = [ACTION_NAMES[a] for a in self.action_buffer]
        terminal = True
        self.action_buffer = []
        self._state = [1.]
        async_reward = self.execute_async(submit_balsam, (config,))

        return self._state, terminal, async_reward

    @property
    def states(self):
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=self.state_space.max_tokens, type='int')

    def copy(self):
        """Return a copy of self."""

        return NasEnvironment(np.copy(self._state), self._score)

    def state(self):
        """Return current state."""
        return self._state

    def score(self):
        """Return current score."""
        return self._score

    def print_state(self):
        print(self.state)

def test_nas_env():
    state_space = StateSpace()
    environment = NasEnvironment(state_space)

if __name__ == '__main__':
    test_nas_env()

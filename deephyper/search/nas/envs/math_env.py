import numpy as np

import gym
from deephyper.benchmarks.benchmark_functions_wrappers import polynome_2
from gym import spaces

f, (a, b), optimum = polynome_2()
DIST_SIZE = 10
NUM_DIM   = 10
VALS      = np.linspace(a, b, DIST_SIZE)

class MathEnv(gym.Env):

    def __init__(self, evaluator):

        self.evaluator = evaluator
        self.observation_space = spaces.Box(low=-0, high=DIST_SIZE-1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(DIST_SIZE)

        self._state = np.array([1.])
        self.action_buffer = []
        self.num_timesteps = NUM_DIM

    def step(self, action, index, rank=None):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            # new_episode = len(self.action_buffer) == 1
            terminal = False
            reward = 0
            self._state = np.array([float(action)])
            return self._state, reward, terminal, {}

        conv_action = [VALS[a] for a in self.action_buffer]

        # new_episode = False
        terminal = True
        self.action_buffer = []
        self._state = np.array([1.])
        cfg = {}
        cfg['x'] = conv_action
        cfg['w'] = index
        if rank != None:
            cfg['rank'] = rank

        self.evaluator.add_eval(cfg)

        # ob, reward, terminal
        return self._state, None, terminal, {}

    def get_rewards_ready(self):
        return self.evaluator.get_finished_evals()

    def reset(self):
        self.__init__(self.evaluator)
        return self._state

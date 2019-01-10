import numpy as np
np.random.seed(22)

import gym
from deephyper.benchmark.benchmark_functions_wrappers import polynome_2, levy_
from gym import spaces

f, (a, b), optimum = polynome_2()
A_MIN = a
B_MAX = b
# f, (a, b), optimum = levy_()
DIST_SIZE = 10 # size of distribution, number of possible actions per timestep
NUM_DIM   = 10 # corresponds to the number of timesteps
VALS      = [np.linspace(a, b, DIST_SIZE) for i in range(NUM_DIM)]
for arr in VALS:
    np.random.shuffle(arr)

class MathEnv(gym.Env):

    def __init__(self, evaluator):

        self.evaluator = evaluator
        self.observation_space = spaces.Box(low=1., high=1., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(DIST_SIZE)

        self._state = np.array([1.])
        self.action_buffer = []
        self.num_timesteps = NUM_DIM

    def step(self, action, index, rank=None):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            terminal = False
            reward = 0
            # self._state = np.array([float(action)])
            # self._state = np.array([ float(VALS[len(self.action_buffer)][action])] )
            self._state = np.array([1.])
            return self._state, reward, terminal, {}

        conv_action = [arr[act] for act, arr in zip(self.action_buffer, VALS)]

        terminal = True
        self._state = np.array([1.])
        cfg = {}
        cfg['arch_seq'] = [int(a)/(DIST_SIZE-1) for a in self.action_buffer]
        self.action_buffer = []
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

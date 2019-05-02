import numpy as np

import gym
from gym import spaces


class NasEnv(gym.Env):

    def __init__(self, space, evaluator, structure):

        self.space = space
        self.structure = structure
        self.evaluator = evaluator

        num_actions = self.structure.max_num_ops
        self.observation_space = spaces.Box(
            low=-0, high=num_actions, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.structure.max_num_ops)

        self._state = np.array([1.])
        self.action_buffer = []
        self.num_timesteps = self.structure.num_nodes

    def step(self, action, index, rank=None):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            # new_episode = len(self.action_buffer) == 1
            terminal = False
            reward = 0
            self._state = np.array([float(action)])
            return self._state, reward, terminal, {}

        conv_action = np.array(self.action_buffer) / self.structure.max_num_ops

        # new_episode = False
        terminal = True
        self.action_buffer = []
        self._state = np.array([1.])

        cfg = self.space.copy()
        cfg['arch_seq'] = list(conv_action)
        cfg['w'] = index
        if rank is not None:
            cfg['rank'] = rank

        self.evaluator.add_eval(cfg)

        # ob, reward, terminal
        return self._state, None, terminal, {}

    def get_rewards_ready(self):
        return self.evaluator.get_finished_evals()

    def reset(self):
        self.__init__(self.space, self.evaluator, self.structure)
        return self._state

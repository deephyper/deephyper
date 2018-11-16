import numpy as np

import gym
from gym import spaces


class NasEnvEmb(gym.Env):

    def __init__(self, space, evaluator, structure):

        self.space = space
        self.structure = structure
        self.evaluator = evaluator

        self.dim_ob_sp = len(structure.get_hash(0, 0))
        self.observation_space = spaces.Box(low=0, high=1,
            shape=(self.dim_ob_sp,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.structure.max_num_ops)

        self._state = np.array([0 for i in range(self.dim_ob_sp)])
        self.action_buffer = []
        self.num_timesteps = self.structure.num_nodes

    def step(self, action, index, rank=None):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            terminal = False
            reward = 0
            action_hash = self.structure.get_hash(len(self.action_buffer)-1, action)
            self._state = np.array([float(e) for e in action_hash])
            return self._state, reward, terminal, {}

        conv_action = np.array(self.action_buffer) / self.structure.max_num_ops

        terminal = True
        self.action_buffer = []
        self._state = np.array([0 for i in range(self.dim_ob_sp)])

        cfg = self.space.copy()
        cfg['arch_seq'] = list(conv_action)
        cfg['w'] = index
        if rank != None:
            cfg['rank'] = rank

        self.evaluator.add_eval(cfg)

        # ob, reward, terminal
        return self._state, None, terminal, {}

    def get_rewards_ready(self):
        return self.evaluator.get_finished_evals()

    def reset(self):
        self.__init__(self.space, self.evaluator, self.structure)
        return self._state

import time

import gym
import numpy as np
from gym import spaces

from deephyper.search import util
from deephyper.search.nas.baselines.common.vec_env import VecEnv
from deephyper.core.logs.logging import JsonMessage as jm

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

dhlogger = util.conf_logger(
    'deephyper.search.nas.env.neural_architecture_envs')


class NeuralArchitectureVecEnv(VecEnv):
    """Multiple environments neural architecture generation. One environment corresponds to one deep neural network architecture.

    Args:
            num_envs (int): number of environments to run in parallel.
            space (dict): neural architecture search space from the Problem.
            evaluator (Evaluator): evaluator to use to evaluate deep neural networks generated.
            structure (KerasStructure): structure to build deep neural networks.
    """

    def __init__(self, num_envs, space, evaluator, structure):
        assert num_envs >= 1

        self.space = space
        self.structure = structure
        self.evaluator = evaluator

        observation_space = spaces.Box(
            low=0,
            high=self.structure.max_num_ops,
            shape=(1,),
            dtype=np.float32)
        action_space = spaces.Discrete(self.structure.max_num_ops)
        super().__init__(num_envs, observation_space, action_space)

        self.action_buffers = [[] for _ in range(self.num_envs)]
        self.states = np.stack([np.array([1.]) for _ in range(self.num_envs)])
        self.num_actions_per_env = self.structure.num_nodes
        self.eval_uids = []
        self.stats = {}

    def step_async(self, actions):
        assert len(actions) == self.num_envs

        # Filling buffers corresponding to each environment
        for i in range(len(actions)):
            self.action_buffers[i].append(actions[i])

        # Submitting evals to balsam when whole sequences are ready
        if len(self.action_buffers[0]) == self.num_actions_per_env:
            XX = []
            for i in range(len(actions)):
                conv_action = np.array(self.action_buffers[i]) / \
                    self.structure.max_num_ops

                cfg = self.space.copy()
                cfg['arch_seq'] = list(conv_action)
                XX.append(cfg)

            self.stats['batch_computation'] = time.time()

            self.eval_uids = self.evaluator.add_eval_batch(XX)

    def step_wait(self):
        obs = [np.array([float(action_seq[-1])])
                for action_seq in self.action_buffers]

        if len(self.action_buffers[0]) < self.num_actions_per_env:
            # Results are already known here...
            rews = [0 for _ in self.action_buffers]
            dones = [False for _ in self.action_buffers]
            infos = {}
        else:
            # Waiting results from balsam
            results = self.evaluator.await_evals(
                self.eval_uids)  # Not blocking
            rews = [rew for cfg, rew in results]  # Blocking generator

            self.stats['batch_computation'] = time.time() - \
                self.stats['batch_computation']
            self.stats['num_cache_used'] = self.evaluator.stats['num_cache_used']
            self.stats['rank'] = MPI.COMM_WORLD.Get_rank(
            ) if MPI is not None else 0

            dones = [True for _ in rews]
            infos = [{
                'episode': {
                    'r': r,
                    'l': self.num_actions_per_env
                } for r in rews}]  # TODO

            self.stats['rewards'] = rews
            self.stats['arch_seq'] = self.action_buffers

            dhlogger.info(jm(type='env_stats', **self.stats))

            self.reset()

        return np.stack(obs), np.array(rews), np.array(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        self.__init__(self.num_envs, self.space,
                      self.evaluator, self.structure)
        self._states = np.stack([np.array([1.]) for _ in range(self.num_envs)])
        return self._states

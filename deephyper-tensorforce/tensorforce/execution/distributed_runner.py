# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.execution.base_runner import BaseRunner
from multiprocessing.pool import ApplyResult

import sys
import time
from six.moves import xrange
import warnings
from inspect import getargspec
from tqdm import tqdm

FREE = 'free'
BUSY = 'busy'
INIT = 'init'

class DistributedRunner(BaseRunner):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0):
        """
        Initialize a single Runner object (one Agent/one Environment).

        Args:
            id_ (int): The ID of this Runner (for distributed TF runs).
        """
        super(DistributedRunner, self).__init__(agent, environment, repeat_actions, history)

        self.id = id_  # the worker's ID in a distributed run (default=0)
        self.running_time = None
        self.num_parallel = self.agent.execution['num_parallel']
        self.workers = None
        print('DistributedRunner with {} workers.'.format(self.num_parallel))

    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None,
            deterministic=False, episode_finished=None, summary_report=None,
            summary_interval=None, testing=False,
            sleep=None):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """
        self.running_time = time.time()

        self.agent.reset()

        if num_episodes is not None:
            num_episodes += self.agent.episode

        if num_timesteps is not None:
            num_timesteps += self.agent.timestep

        update_mode = self.agent.update_mode # update_mode['batch_size']

        # add progress bar
        pbar = tqdm(total=num_episodes)

        # episode loop
        self.global_episode = 0

        self.workers = workers = {FREE: [i for i in range(self.num_parallel)], BUSY:[], INIT:[]}
        episodes_rewards = [ None for _ in range(self.num_parallel)]
        # print(f'init workers: {workers}')

        while True:

            # Launch free workers
            for _ in range(len(self.workers[FREE])):
                # get a free worker and change its state
                w = workers[FREE].pop()
                workers[INIT].append(w)

                #init worker: w
                episodes_rewards[w] = list()
                state = self.environment.reset()
                self.agent.reset()

                # time step (within episode) loop
                for i in range(self.environment.num_timesteps):
                    action = self.agent.act(states=state,
                                            deterministic=deterministic,
                                            index=w)

                    state, terminal, reward = self.environment.execute(action=action)

                    episodes_rewards[w].append(reward)
                    if not isinstance(reward, ApplyResult):
                        self.agent.observe(terminal=terminal, reward=reward, index=w)

                # change state of workers
                workers[INIT].remove(w)
                workers[BUSY].append(w)

            for w in workers[BUSY][:]:

                try:
                    if (episodes_rewards[w][-1].successful()):

                        reward = episodes_rewards[w][-1].get()
                        self.episode_rewards.append(reward)
                        # print(f'worker {w}, reward: {reward}')
                        self.agent.observe(terminal=True, reward=reward, index=w)
                        workers[BUSY].remove(w)
                        workers[FREE].append(w)

                        self.global_episode += 1
                        pbar.update(1)

                        if episode_finished is not None:
                            episode_finished(self)

                except AssertionError:
                    # print(f'worker {w} is still busy !')
                    pass

            if (num_episodes is not None and self.global_episode >= num_episodes):
                break

        pbar.close()

        self.running_time = time.time() - self.running_time

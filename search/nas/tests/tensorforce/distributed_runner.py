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
        self.current_timestep = None  # the time step in the current episode
        self.running_time = None

    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None,
            deterministic=False, episode_finished=None, summary_report=None,
            summary_interval=None, timesteps=None, episodes=None, testing=False,
            sleep=None):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """
        self.running_time = time.time()

        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps
            warnings.warn("WARNING: `timesteps` parameter is deprecated, use `num_timesteps` instead.",
                          category=DeprecationWarning)
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        if num_episodes is not None:
            num_episodes += self.agent.episode

        if num_timesteps is not None:
            num_timesteps += self.agent.timestep

        update_mode = self.agent.update_mode # update_mode['batch_size']

        # add progress bar
        with tqdm(total=num_episodes) as pbar:

            # episode loop
            self.global_episode = 0
            while True:

                episodes_start_time = []
                episodes_rewards = [[] for _ in range(update_mode['batch_size'])]

                for episode_i in range(update_mode['batch_size']):
                    episodes_start_time.append(time.time())
                    state = self.environment.reset()
                    self.agent.reset()


                    # time step (within episode) loop
                    for _ in range(self.environment.num_timesteps):
                        action = self.agent.act(states=state, deterministic=deterministic)

                        state, terminal, reward = self.environment.execute(action=action)

                        episodes_rewards[episode_i].append(reward)

                for episode_i in range(update_mode['batch_size']):

                    # Update global counters.
                    self.global_timestep = 0

                    self.episode_rewards.append(0)
                    self.current_timestep = 0

                    for i, reward in enumerate(episodes_rewards[episode_i]):
                        terminal = (i == (self.environment.num_timesteps - 1))
                        if not testing:
                            if isinstance(reward, ApplyResult):
                                reward = reward.get()
                            self.agent.observe(terminal=terminal, reward=reward)
                            self.episode_rewards[-1] += reward

                        self.global_timestep += 1
                        self.current_timestep += 1

                    # Update our episode stats.
                    time_passed = time.time() - episodes_start_time[episode_i]
                    self.episode_timesteps.append(self.current_timestep)
                    self.episode_times.append(time_passed)

                    self.global_episode += 1
                    pbar.update(1)

                if (num_episodes is not None and self.global_episode >= num_episodes):
                    break
            pbar.update(num_episodes - self.global_episode)

        self.running_time = time.time() - self.running_time

    # keep backwards compatibility
    @property
    def episode_timestep(self):
        return self.current_timestep


# more descriptive alias for Runner class
# DistributedTFRunner = DistributedRunner
# SingleRunner = DistributedRunner

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

import numpy as np
import matplotlib.pyplot as plt

from tensorforce.agents import VPGAgent
from tensorforce.execution import Runner
# from tensorforce.contrib.openai_gym import OpenAIGymA
from math_fun import MathFun

# Create an OpenAIgym environment.
# environment = OpenAIGym('CartPole-v0', visualize=True)
environment = MathFun()

# Network as list of layers
# - Embedding layer:
#   - For Gym environments utilizing a discrete observation space, an
#     "embedding" layer should be inserted at the head of the network spec.
#     Such environments are usually identified by either:
#     - class ...Env(discrete.DiscreteEnv):
#     - self.observation_space = spaces.Discrete(...)

# Note that depending on the following layers used, the embedding layer *may* need a
# flattening layer

network_spec = [
    # dict(type='embedding', indices=100, size=32),
    # dict(type'flatten'),
    # dict(type='internal_lstm', size=32),
    dict(type='internal_lstm', size=32)
]

agent = VPGAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=20,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    # PPOAgent
    # optimizer=dict(
    #     type='adam',
    #     learning_rate=1e-3
    # ),
    # subsampling_fraction=0.2,
    # optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

# Create the runner
# runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
    return True


def go_learning():
    runner = Runner(agent=agent, environment=environment)
    runner.run(episodes=1000, max_episode_timesteps=200, episode_finished=episode_finished)
    runner.close()
    return runner.episode_rewards

# Print statistics
# print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
#     ep=runner.episode,
#     ar=np.mean(runner.episode_rewards[-100:]))
# )

def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

if __name__ == '__main__':
    episode_rewards = go_learning()
    x = [i for i in range(len(episode_rewards))]
    plt.plot(x, episode_rewards)
    plt.plot(x, max_list(episode_rewards))
    plt.show()

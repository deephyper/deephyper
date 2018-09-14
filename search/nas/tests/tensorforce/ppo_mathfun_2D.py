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
import time
import tensorflow as tf
import argparse

tf.set_random_seed(1000003)
np.random.seed(1000003)

from tensorforce.agents import PPOAgent
from action_history_runner import Runner
from math_fun import MathFun
from benchmark_functions_wrappers import *

# math_func, _, _ = ackley_()
# math_func, _, _ = polynome_2()
# math_func, _, _ = dixonprice_()
# math_func, _, _ = levy_()
# math_func, _, _ = griewank_()

def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('func_wrapper_name', type=str)
    return parser

def get_agent_environment(fw=ackley_):

    # Create an OpenAIgym environment.
    environment = MathFun(num_dim=2, num_action=100, func_wrapper=fw)

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
        dict(type='internal_lstm', size=32)
    ]

    agent = PPOAgent(
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
            batch_size=10,
            # Every 10 episodes
            frequency=1
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
        likelihood_ratio_clipping=0.2,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        subsampling_fraction=0.2,
        optimization_steps=25,
        execution=dict(
            type='single',
            session_config=None,
            distributed_spec=None
        )
    )
    return agent, environment


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
    return True


def go_learning(func_wrapper=ackley_):
    agent, environment = get_agent_environment(func_wrapper)
    runner = Runner(agent=agent, environment=environment)
    t = time.time()
    runner.run(episodes=1000, max_episode_timesteps=200, episode_finished=None)
    t = time.time() - t
    runner.close()
    print(f'Running time: {t}')
    print(f'Awerage reward of last 100 episodes: {np.mean(runner.episode_rewards[-100:])}')
    return runner.episode_rewards, runner


def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl


def create_grid(x, func):
    num_val = len(x)
    x1, x2 = np.meshgrid(x, x)
    z = np.zeros((num_val, num_val))
    for i in range(num_val):
        for j in range(num_val):
            z[i,j] = func([x1[i,j], x2[i,j]])
    return x1, x2, z

def save_plot(name):
    plt.savefig(name+'.png', dpi=300)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    dict_wrappers = {}
    dict_wrappers[ackley_.__name__] = ackley_
    dict_wrappers[polynome_2.__name__] = polynome_2
    dict_wrappers[dixonprice_.__name__] = dixonprice_
    dict_wrappers[griewank_.__name__] = griewank_
    dict_wrappers[levy_.__name__] = levy_
    key = args.func_wrapper_name
    print(f'Go for : {key}')
    fw = dict_wrappers[key]

    episode_rewards, runner = go_learning(fw)
    func_wrapper = runner.environment.func_wrapper
    num_action = runner.environment.num_action
    f, (a, b), optimum = func_wrapper()

    plt.figure(figsize=(16, 10), dpi=300, facecolor='w', edgecolor='k')

    # First figure
    x = [i for i in range(len(episode_rewards))]
    plt.subplot(211)
    plt.title(f'Best reward find so far with PPO on {func_wrapper.__name__} with {num_action} actions per axis')
    plt.plot(x, episode_rewards, label='raw iterations')
    plt.plot(x, max_list(episode_rewards), label='best reward')
    plt.legend()

    x1, x2, z = create_grid(runner.environment.action_tokens, f)
    pts = np.array(runner.episode_actions)
    len_pts = np.shape(pts)[0]

    # Second figure
    plt.subplot(212)
    plt.title(f'Contour plot with PPO on {func_wrapper.__name__} with {num_action} actions per axis')
    plt.contourf(x1, x2, z, 20, cmap='RdGy')
    plt.colorbar()

    optimum = optimum(2)
    plt.plot(optimum[0], optimum[1], 'co')
    # plt.plot(x1, x2, 'g+')

    plt.scatter(pts[:, 0],
                pts[:, 1],
                c=np.array([i for i in range(len_pts)]),
                cmap='hot', vmin=1, vmax=len_pts)
    plt.plot(pts[-1, 0],
             pts[-1, 1], 'bo')
    plt.colorbar()

    loc = '/Users/Deathn0t/Documents/Argonne/deephyper/search/nas/tests/tensorforce/graphs/2dim/100actions_10_1'
    save_plot(f'{loc}/ppo_{func_wrapper.__name__}_2dim_{num_action}actions')

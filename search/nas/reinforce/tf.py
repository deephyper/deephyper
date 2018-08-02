'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 14:59:13
'''
import os
import sys
import tensorflow as tf
import numpy as np
import random

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV3
from deephyper.model.arch import StateSpace

class BasicReinforce:
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 num_features,
                 state_space=None,
                 #division_rate=100.0,
                 division_rate=1.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.3):
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.global_step = global_step
        self.num_features = num_features
        self.state_space = state_space

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, state, num_layers):
        rnn_res = np.array(self.sess.run(self.policy_outputs[:len(state[0])],
                           {self.states: state})).flatten().tolist()
        return self.state_space.parse_state(rnn_res, num_layers)

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(
                tf.float32, [None, None], name="states")

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                _, self.policy_outputs = self.policy_network.get(
                    self.states, self.max_layers)

        # regularization loss
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(
                tf.float32, (None,), name="discounted_rewards")

            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs, _ = self.policy_network.get(
                    self.states, self.max_layers)

            # Here you get all the possible outputs, but be careful there are not always
            # computed
            self.logprobs = self.logprobs[:, -1, :]

            # You need the slice as a tensor because you are building a computational
            # graph (tf.slice). To build this slice, you need to get the dynamic shape of
            # the states as a tensor. States is a tensor. (tf.shape)
            self.logprobs = tf.slice(self.logprobs, [0, 0], tf.shape(self.states))

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logprobs, labels=self.states)
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss = tf.reduce_sum([tf.reduce_sum(
                tf.square(x)) for x in policy_network_variables])  # Regularization
            self.loss = self.pg_loss + self.reg_param * self.reg_loss

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step)

    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        rewards = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss],
                              {self.states: states,
                               self.discounted_rewards: rewards})
        return ls


class BasicReinforceV2:
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 num_features,
                 state_space=None,
                 division_rate=1.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.3):
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.global_step = global_step
        self.num_features = num_features
        self.state_space = state_space

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_actions(self, input_rnn, num_layers):
        num_tokens = num_layers * self.state_space.size
        return self.sess.run(self.policy_outputs_list[:self.num_tokens],
                             {self.input_rnn: input_rnn})
    def create_variables(self):
        self.input_rnn = tf.placeholder(
            tf.float32, [None, 1], name='input_rnn')

        self.num_tokens_tensor = tf.placeholder(
            tf.int32, [1], name='num_tokens')

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                self.logits, self.policy_outputs_list = self.policy_network.get(
                    self.input_rnn, self.max_layers)
                self.policy_outputs = tf.concat(self.policy_outputs_list, axis=0)

        # regularization loss
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(
                tf.float32, (None,), name="discounted_rewards")

            # Here you get all the possible outputs, but be careful there are not always
            # computed: logprobs

            # You need the slice as a tensor because you are building a computational
            # graph (tf.slice). To build this slice, you need to get the dynamic shape of
            # the states as a tensor. States is a tensor. (tf.shape)
            input_rnn_shape = tf.shape(self.input_rnn)
            # shape : [num_tokens, batch_size, num_classes]
            self.logits_slice = tf.slice(self.logprobs, [0, 0, 0],
                                   [self.num_tokens_tensor[0],
                                    input_rnn_shape[0],
                                    self.policy_network.max_num_classes])

            # shape : [num_tokens, batch_size]
            policy_outputs_shape = tf.shape(self.policy_outputs)
            self.policy_outputs_slice = tf.slice(self.policy_outputs, [0, 0],
                                                [self.num_tokens_tensor[0],
                                                policy_outputs_shape[1]])

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_slice, labels=self.policy_outputs_slice)
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss = tf.reduce_sum([tf.reduce_sum(
                tf.square(x)) for x in policy_network_variables])  # Regularization
            self.loss = self.pg_loss + self.reg_param * self.reg_loss

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step)

    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        rewards = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss],
                              {self.input_rnn: states,
                               self.discounted_rewards: rewards})
        return ls


def test_BasicReinforce():
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    state_space.add_state('filter_size', [10., 20., 30.])
    state_space.add_state('drop_out', [])
    state_space.add_state('num_filters', [32., 64.])
    state_space.add_state('skip_conn', [])
    policy_network = NASCellPolicyV3(state_space)
    max_layers = 3

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforce(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                global_step,
                                num_features=state_space.size,
                                state_space=state_space)
    state_l2 = [[10., 0.5, 32., 1.,
                 10., 0.5, 32., 1., 1.]]
    num_layers = 2
    action = reinforce.get_action(state_l2, num_layers)
    print(f'action = {action}')
    reward = 90.
    reinforce.storeRollout([action], reward)
    reinforce.train_step(1)

    state_l3 = [[10., 0.5, 32., 1.,
                 10., 0.5, 32., 1., 1.,
                 10., 0.5, 32., 1., 1., 1.]]
    num_layers = 3
    action = reinforce.get_action(state_l3, num_layers)
    print(f'action = {action}')
    reward = 90.
    reinforce.storeRollout([action], reward)
    reinforce.train_step(1)

if __name__ == '__main__':
    test_BasicReinforce()

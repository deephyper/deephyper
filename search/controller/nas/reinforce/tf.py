'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 14:59:13
'''
import tensorflow as tf
import numpy as np
import random


class BasicReinforce:
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 num_features,
                 division_rate=100.0,
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

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, state):
        return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(
                tf.float32, [None, self.max_layers*self.num_features], name="states")

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                self.policy_outputs = self.policy_network.get(
                    self.states, self.max_layers)

            self.action_scores = tf.identity(
                self.policy_outputs, name="action_scores")

            self.predicted_action = tf.cast(tf.scalar_mul(
                self.division_rate, self.action_scores), tf.int32, name="predicted_action")

        # regularization loss
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(
                tf.float32, (None,), name="discounted_rewards")

            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network.get(
                    self.states, self.max_layers)

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logprobs[:, -1, :], labels=self.states)
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

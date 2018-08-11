'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 14:59:13
'''
import os
import sys
import tensorflow as tf
import numpy as np
import random
import math

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

from deephyper.search.nas.policy.tf import NASCellPolicyV5
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
            print(f'cross entropy shape: {self.cross_entropy_loss.shape}')
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

class BasicReinforceV5:
    def __init__(self, sess, optimizer, policy_network, max_layers,
                batch_size,
                global_step,
                state_space=None,
                division_rate=1.0,
                reg_param=0.001,
                discount_factor=0.99,
                exploration=1.,
                 exploration_decay = 0.05):
        '''
        Args
            sess: tensorflow session
            optimizer: optimizer used for controller
            policy_network: class of policy used by the controller
            max_layers: maximum number of layers generated by the controller
            batch_size: number of models generated by the controller
            global_step:
            state_space: search space definition used by the controller
            division_rate:
            reg_param:
            discount_factor:
            exploration:
        '''
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.batch_size = batch_size
        self.global_step = global_step
        self.state_space = state_space
        self.exploration= exploration
        self.exploration_decay = exploration_decay

        self.max_reward = 0
        self.reward_list = [] # a reward is added only one time
        self.reward_buffer = [] # a reward is duplicate num_tokens time, corresponding to a 1D list with batch_size rewards num_tokens_in_one_batch time
        self.state_buffer = []
        self.rewards_b = 0

        self.create_variables()
        self.policy_network.restore_model(sess = self.sess)
        init = tf.global_variables_initializer()
        sess.run(init)

    def get_actions(self, rnn_input, num_layers):
        '''
            Generations a list of index corresponding to the actions choosed by the controller.
            Args
                rnn_input: list of shape (batch_size) input of the rnn
                num_layers: int
        '''
        if self.state_space.feature_is_defined('skip_conn'):
            num_tokens_for_one = ((self.state_space.size - 1) * num_layers + num_layers * (num_layers - 1) // 2)
        else:
            num_tokens_for_one = self.state_space.size * num_layers
        self.num_tokens = num_tokens_for_one * self.batch_size

        if random.random() > self.exploration:
            policy_outputs_ =  self.sess.run(
                self.policy_outputs[:self.num_tokens],
                {self.rnn_input:rnn_input,
                self.num_tokens_tensor: [self.num_tokens]})

            return policy_outputs_
        else:
            tokens = [int(random.random() * self.policy_network.max_num_classes) * 1. for _ in range(self.num_tokens)]
            return tokens

    def create_variables(self):
        self.rnn_input = tf.placeholder(
            dtype=tf.float32, shape=(self.batch_size), name='rnn_input')

        self.num_tokens_tensor = tf.placeholder(
            tf.int32, [1], name='num_tokens')

        self.batch_labels = tf.placeholder(tf.int32,[None,self.batch_size], name='labels')

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                self.policy_outputs, self.before_softmax_outputs = self.policy_network.get(
                    self.rnn_input, self.max_layers)

            self.action_scores = tf.identity(
                self.policy_outputs, name="action_scores")

            self.predicted_action = tf.cast(
                tf.scalar_mul(
                    self.division_rate,
                    self.action_scores),
                    tf.float32,
                    name="predicted_action")

        # regularization loss
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(
                tf.float32,
                shape=(None),
                name="discounted_rewards")

            self.policy_outputs_slice = tf.slice(
                self.policy_outputs,
                [0],
                [self.num_tokens_tensor[0]])

            self.before_softmax_outputs_slice = tf.slice(
                self.before_softmax_outputs,
                [0, 0, 0],
                [self.num_tokens_tensor[0] // self.batch_size,
                self.batch_size,
                self.policy_network.max_num_classes])

            self.cross_entropy_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.before_softmax_outputs_slice,
                labels=self.batch_labels)
            self.cross_entropy_loss = tf.multiply(
                self.cross_entropy_loss_,
                self.discounted_rewards)

            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss = tf.reduce_sum([tf.reduce_sum(
                tf.square(x)) for x in policy_network_variables])  # Regularization
            self.loss = self.pg_loss + self.reg_param * self.reg_loss

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    #self.gradients[i] = (tf.multiply(grad, tf.reduce_mean(self.discounted_rewards)), var)
                    #self.gradients[i] = (tf.scalar_mul(self.discounted_rewards, grad), var)
                    self.gradients[i] = (grad, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step)

    def storeRollout(self, state, rewards, num_layers):
        '''
        Args
            state:
            rewards: a list
            num_layers: an int
        '''
        if self.state_space.feature_is_defined('skip_conn'):
            num_tokens_for_one = ((self.state_space.size - 1) * num_layers + num_layers * (num_layers - 1) // 2)
        else:
            num_tokens_for_one = self.state_space.size * num_layers
        self.num_tokens = num_tokens_for_one * self.batch_size

        self.reward_list.extend(rewards)
        self.max_reward = max(self.max_reward, max(rewards))
        for i in range(0, self.num_tokens, self.batch_size):
            self.reward_buffer.extend(rewards)
            self.state_buffer.extend(state[-i-self.batch_size:][:self.batch_size])
            #rewards = [self.discount_factor * x for x in rewards]

    def train_step(self, num_layers, init_state):
        if self.state_space.feature_is_defined('skip_conn'):
            num_tokens_for_one = ((self.state_space.size - 1) * num_layers + num_layers * (num_layers - 1) // 2)
        else:
            num_tokens_for_one = self.state_space.size * num_layers
        self.num_tokens = num_tokens_for_one * self.batch_size

        steps_count = self.num_tokens
        states = np.reshape(np.array(self.state_buffer[-steps_count:]),
        (self.num_tokens // self.batch_size, self.batch_size)) / self.division_rate
        prev_state = init_state
        rewards = self.reward_buffer[-steps_count:]
        self.rewards_b = ema(self.reward_list, 100)
        # print(f'buffer: {self.reward_buffer}')
        #self.reward_b = max(self.reward_buffer) if (len(self.reward_buffer) > 1) else 0
        rewards = [x-self.rewards_b for x in rewards]

        _, ls = self.sess.run([self.train_op,
                                          self.loss,],
                                   {self.rnn_input: prev_state,
                                    self.discounted_rewards: rewards,
                                    self.batch_labels: states,
                                    self.num_tokens_tensor: [self.num_tokens]})
        self.policy_network.save_model(self.sess)
        self.exploration *= math.exp(-self.exploration_decay)
        return ls

def sma(data, window):
    """
    Calculates Simple Moving Average
    http://fxtrade.oanda.com/learn/forex-indicators/simple-moving-average
    """
    if len(data) < window:
        return None
    return sum(data[-window:]) / float(window)

def ema(data, window):
    if len(data) < 2 * window:
        #return sum(data)/len(data)
        return max(data)
    c = 2.0 / (window + 1)
    current_ema = sma(data[-window*2:-window], window)
    for value in data[-window:]:
        current_ema = (c * value) + ((1 - c) * current_ema)
    return current_ema

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

def test_BasicReinforce5():
    session = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    state_space = StateSpace()
    state_space.add_state('filter_size', [10., 20., 30.])
    #state_space.add_state('drop_out', [])
    state_space.add_state('num_filters', [32., 64.])
    state_space.add_state('skip_conn', [])
    policy_network = NASCellPolicyV5(state_space, save_path='savepoint/model')
    max_layers = 8
    batch_size = 2

    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                                500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # for the CONTROLLER
    reinforce = BasicReinforceV5(session,
                                optimizer,
                                policy_network,
                                max_layers,
                                batch_size,
                                global_step,
                                state_space=state_space)
    #state_l2 = [[10., 0.5, 32., 1.,
    #             10., 0.5, 32., 1., 1.]]
    init_seeds = [1. * i / batch_size for i in range(batch_size)]
    #init_seeds = [.5, .4, -.1, -.7]
    for num_layers in range(1, max_layers):
        actions = reinforce.get_actions(init_seeds, num_layers)
        rewards = [.90] * batch_size
        print(f' num_layer: {num_layers} action = {actions} rewards = {rewards}')
        reinforce.storeRollout(actions, rewards, num_layers)
        reinforce.train_step(num_layers, init_seeds)


if __name__ == '__main__':
    test_BasicReinforce5()

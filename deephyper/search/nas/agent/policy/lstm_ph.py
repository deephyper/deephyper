import tensorflow as tf
import numpy as np

import  deephyper.search.nas.utils.common.tf_util as U
import gym
from  deephyper.search.nas.utils.common.distributions import make_pdtype
from  deephyper.search.nas.utils.common.mpi_running_mean_std import \
    RunningMeanStd


class LstmPolicy(object):
    recurrent = True
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, num_units, gaussian_fixed_var=True, async_update=False):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space) # pd: probability distribution
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape, use_mpi=(not async_update))

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            lstm = tf.contrib.rnn.LSTMCell(
                num_units=num_units,
                name=f'rnn_cell_vf',
                initializer=U.normc_initializer(1.0))

            init_c, init_h = lstm.zero_state(1, dtype=tf.float32)

            self.input_c_vf = U.get_placeholder(dtype=tf.float32, name="c_vf", shape=[None]+list(init_c.get_shape()[1:]))
            self.input_h_vf = U.get_placeholder(dtype=tf.float32, name="h_vf", shape=[None]+list(init_h.get_shape()[1:]))

            inpt_vf = tf.expand_dims(obz, 0)
            out_vf, (new_c, new_h) = tf.nn.dynamic_rnn(lstm,
                inpt_vf,
                initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.input_c_vf, self.input_h_vf),
                dtype=tf.float32)
            out_vf = tf.squeeze(out_vf, axis=[0])

            self.vpred = tf.layers.dense(out_vf, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
            self.out_hs_vf = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        with tf.variable_scope('pol'):

            lstm = tf.contrib.rnn.LSTMCell(
                num_units=num_units,
                name=f'rnn_cell_pol',
                initializer=U.normc_initializer(1.0))

            init_c, init_h = lstm.zero_state(1, dtype=tf.float32)

            self.input_c_pol = U.get_placeholder(dtype=tf.float32, name="c_pol", shape=[None]+list(init_c.get_shape()[1:]))
            self.input_h_pol = U.get_placeholder(dtype=tf.float32, name="h_pol", shape=[None]+list(init_h.get_shape()[1:]))

            inpt_pol = tf.expand_dims(obz, 0)
            out_pol, (new_c, new_h) = tf.nn.dynamic_rnn(lstm,
                inpt_pol,
                initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.input_c_vf, self.input_h_vf),
                dtype=tf.float32)
            out_pol = tf.squeeze(out_pol, axis=[0])
            self.out_hs_pol = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(out_pol, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(out_pol, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, self.input_c_vf, self.input_h_vf, self.input_c_pol, self.input_h_pol], 
                               [ac, self.vpred, self.out_hs_vf, self.out_hs_pol])

    def act(self, stochastic, ob, c_vf, h_vf, c_pol, h_pol):
        """Action of policy
        
        Arguments:
            stochastic {[type]} -- [description]
            ob {[type]} -- observation from the environment
            c_vf {[type]} -- hidden state of the lstm for value function
            h_vf {[type]} -- hidden state of the lstm for value function
            c_pol {[type]} -- hidden state of the lstm for policy
            h_pol {[type]} -- hidden state of the lstm for policy
        """
        
        ac1, vpred1, new_hs_vf, new_hs_pol =  self._act(stochastic, ob[None], c_vf, h_vf, c_pol, h_pol)
        return ac1[0], vpred1[0], new_hs_vf, new_hs_pol

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

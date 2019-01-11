import tensorflow as tf

import  deephyper.search.nas.utils.common.tf_util as U
import gym
from  deephyper.search.nas.utils.common.distributions import make_pdtype
from  deephyper.search.nas.utils.common.mpi_running_mean_std import \
    RunningMeanStd


class LstmPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, num_units, gaussian_fixed_var=True, async_update=False):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        full_path_is_done = tf.get_variable("full_path_is_done", dtype=tf.bool,
                initializer=True, trainable=False)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape, use_mpi=(not async_update))

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz

            lstm = tf.contrib.rnn.LSTMCell(
                num_units=num_units,
                name=f'rnn_cell',
                initializer=U.normc_initializer(1.0))

            init_lstm_state = lstm.zero_state(1, dtype=tf.float32)
            init_t = tf.get_variable('init_t', dtype=tf.float32, initializer=init_lstm_state, trainable=False)

            v_lstm_state = tf.get_variable("v_lstm_state", dtype=tf.float32,
                initializer=init_lstm_state, trainable=False)

            ba_state = tf.get_variable("ba_state", dtype=tf.float32,
                initializer=init_lstm_state, trainable=False)

            assign_ba_state = tf.cond(full_path_is_done,
                lambda: tf.assign(ba_state, init_t), # TRUE
                lambda: tf.assign(ba_state, ba_state)) # FALSE

            lstm_state = tf.cond(tf.equal(tf.shape(ob)[0], 1),
                lambda: v_lstm_state,
                lambda: ba_state)

            assign_fpid = tf.assign(full_path_is_done, tf.math.greater(tf.shape(ob)[0], 1))

            with tf.control_dependencies([assign_ba_state]):
                last_out = tf.expand_dims(last_out, 0)
                last_out, lstm_new_state = tf.nn.dynamic_rnn(lstm,
                    last_out,
                    initial_state=init_lstm_state,
                    dtype=tf.float32)
                assign_new_state = tf.assign(v_lstm_state, lstm_new_state)
                last_out = tf.squeeze(last_out, axis=[0])

            with tf.control_dependencies([assign_new_state, assign_fpid]):
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz

            lstm = tf.contrib.rnn.LSTMCell(
                num_units=num_units,
                name=f'rnn_cell',
                initializer=U.normc_initializer(1.0),
                state_is_tuple=False)

            init_lstm_state = lstm.zero_state(1, dtype=tf.float32)
            init_t = tf.get_variable('init_t', dtype=tf.float32, initializer=init_lstm_state, trainable=False)

            v_lstm_state = tf.get_variable("v_lstm_state", dtype=tf.float32,
                initializer=init_lstm_state, trainable=False)

            ba_state = tf.get_variable("ba_state", dtype=tf.float32,
                initializer=init_lstm_state, trainable=False)

            assign_ba_state = tf.cond(full_path_is_done,
                lambda: tf.assign(ba_state, init_t), # TRUE
                lambda: tf.assign(ba_state, ba_state)) # FALSE

            lstm_state = tf.cond(tf.equal(tf.shape(ob)[0], 1),
                lambda: v_lstm_state,
                lambda: ba_state)

            assign_fpid = tf.assign(full_path_is_done, tf.math.greater(tf.shape(ob)[0], 1))

            with tf.control_dependencies([assign_ba_state]):
                last_out = tf.expand_dims(last_out, 0)
                last_out, lstm_new_state = tf.nn.dynamic_rnn(lstm,
                    last_out,
                    initial_state=lstm_state,
                    dtype=tf.float32)
                assign_new_state = tf.assign(v_lstm_state, lstm_new_state)
                last_out = tf.squeeze(last_out, axis=[0])


            with tf.control_dependencies([assign_new_state, assign_fpid]):
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

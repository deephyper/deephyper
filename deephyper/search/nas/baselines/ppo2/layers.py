import numpy as np
import tensorflow as tf

from deephyper.search.nas.baselines.a2c.utils import ortho_init, lstm, lnlstm
from deephyper.search.nas.baselines.common.models import register, nature_cnn


class RNN(object):
    def __init__(self, func, memory_size=None):
        self._func = func
        self.memory_size = memory_size

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


@register("ppo_lstm")
def ppo_lstm(num_units=128, layer_norm=False):
    def network_fn(input, mask, state):
        input = tf.layers.flatten(input)
        mask = tf.to_float(mask)

        if layer_norm:
            h, next_state = lnlstm(
                [input], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([input], [mask[:, None]],
                                 state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(network_fn, memory_size=num_units * 2)


@register("ppo_lstm_128")
def ppo_lstm_128():
    return ppo_lstm(num_units=128)


@register("ppo_lnlstm_128")
def ppo_lnlstm_128():
    return ppo_lstm(num_units=128, layer_norm=True)


@register("ppo_lstm_64")
def ppo_lstm_64():
    return ppo_lstm(num_units=64)


@register("ppo_lnlstm_64")
def ppo_lnlstm_64():
    return ppo_lstm(num_units=64, layer_norm=True)


@register("ppo_lstm_32")
def ppo_lstm_32():
    return ppo_lstm(num_units=32)


@register("ppo_lnlstm_32")
def ppo_lnlstm_32():
    return ppo_lstm(num_units=32, layer_norm=True)


@register("ppo_cnn_lstm")
def ppo_cnn_lstm(num_units=128, layer_norm=False, **conv_kwargs):
    def network_fn(input, mask, state):
        mask = tf.to_float(mask)
        initializer = ortho_init(np.sqrt(2))

        h = nature_cnn(input, **conv_kwargs)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, units=512, activation=tf.nn.relu,
                            kernel_initializer=initializer)

        if layer_norm:
            h, next_state = lnlstm(
                [h], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([h], [mask[:, None]],
                                 state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(network_fn, memory_size=num_units * 2)


@register("ppo_cnn_lnlstm")
def ppo_cnn_lnlstm(num_units=128, **conv_kwargs):
    return ppo_cnn_lstm(num_units, layer_norm=True, **conv_kwargs)

"""
https://arxiv.org/pdf/2004.02967.pdf
"""
import tensorflow as tf


def group_std_2d(x, groups=32, eps=1e-5):
    N, H, W, C = tf.shape(x)
    x = tf.reshape(x, [N, H, W, groups, C // groups])
    _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    std = tf.sqrt(var + eps)
    std = tf.broadcast_to(std, x.shape)
    return tf.reshape(std, (N, H, W, C))


def group_std_1d(x, eps=1e-5):
    m, n = tf.shape(x)
    _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    std = tf.sqrt(var + eps)
    std = tf.broadcast_to(std, x.shape)
    return tf.reshape(std, (m, n))


class EvoNorm2dS0(tf.keras.layers.Layer):
    def __init__(self, in_channels, groups=32, nonlinear=True):
        super(EvoNorm2dS0, self).__init__()
        self.nonlinear = nonlinear
        self.groups = groups

        def build(self):
            self.gamma = self.add_variable(
                "gamma",
                shape=(1, 1, 1, self.in_channels),
                initializer=tf.initializers.Ones(),
            )
            self.beta = self.add_variable(
                "beta",
                shape=(1, 1, 1, self.in_channels),
                initializer=tf.initializers.Zeros(),
            )
            if self.nonlinear:
                self.v = self.add_variable(
                    "v",
                    shape=(1, 1, 1, self.in_channels),
                    initializer=tf.initializers.Ones(),
                )

        def call(self, x):
            if self.nonlinear:
                num = x * tf.nn.sigmoid(self.v * x)
                return num / group_std_2d(x) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class EvoNormS01D(tf.keras.layers.Layer):
    def __init__(self, nonlinear=True):
        super(EvoNormS01D, self).__init__()
        self.nonlinear = nonlinear

        def build(self):
            self.gamma = self.add_variable(
                "gamma", shape=(1, 1), initializer=tf.keras.initializers.Ones()
            )
            self.beta = self.add_variable(
                "beta", shape=(1, 1), initializer=tf.keras.initializers.Zeros()
            )
            if self.nonlinear:
                self.v = self.add_variable(
                    "v", shape=(1, 1), initializer=tf.keras.initializers.Ones()
                )

        def call(self, x):
            if self.nonlinear:
                num = x * tf.nn.sigmoid(self.v * x)
                return num / group_std_1d(x) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta

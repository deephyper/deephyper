import tensorflow as tf

from deephyper.search.nas.operation.basic import Operation


class MLP(Operation):
    def __init__(self, layers, units, activation):
        self.layers = layers
        self.units = units
        self.activation = activation

    def __str__(self):
        return f'MLP_{self.layers}x{self.units}_{self.activation.__name__}'

    def __call__(self, inputs, **kwargs):
        out = inputs[0]
        for _ in range(self.layers):
            out = tf.layers.dense(
                        out,
                        units=self.units,
                        activation=self.activation,
                        kernel_initializer=tf.initializers.random_uniform()
                    )
        return out


MLP_2x2_relu = MLP(2, 2, tf.nn.relu)
MLP_3x3_relu = MLP(3, 3, tf.nn.relu)
MLP_5x5_relu = MLP(5, 5, tf.nn.relu)
MLP_2x2_tanh = MLP(2, 2, tf.tanh)
MLP_3x3_tanh = MLP(3, 3, tf.tanh)
MLP_5x5_tanh = MLP(5, 5, tf.tanh)
mlp_ops = [MLP_2x2_relu,
           MLP_3x3_relu,
           MLP_5x5_relu,
           MLP_2x2_tanh,
           MLP_3x3_tanh,
           MLP_5x5_tanh]


class Dropout(Operation):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def __str__(self):
        return f'Dropout({int(self.keep_prob*100)})'

    def __call__(self, inputs, **kwargs):
        inpt = inputs[0]
        if kwargs.get('train'):
            out = tf.nn.dropout(inpt, self.keep_prob)
        else:
            out = tf.nn.dropout(inpt, 1.)
        return out


dropout_ops = [Dropout(1.),
               Dropout(0.9),
               Dropout(0.8),
               Dropout(0.7),
               Dropout(0.6),
               Dropout(0.5),
               Dropout(0.4)]

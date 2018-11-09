import tensorflow as tf

from deephyper.searches.nas.operation.basic import Operation


class MLP(Operation):
    """Multi Layer Perceptron operation.

    Help you to create a perceptron with n layers, m units per layer and an activation function.

    Args:
        layers (int): number of layers.
        units (int): number of units per layer.
        activation: an activation function from tensorflow.
    """
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


class Dropout(Operation):
    """Dropout operation.

    Help you to create a dropout operation.

    Args:
        keep_prob (float): probability to keep a value from the input.
    """
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

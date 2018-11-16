import tensorflow as tf
from tensorflow import keras
import numpy as np

from deephyper.search.nas.model.space.op.basic import Operation

class Concatenate(Operation):
    """Concatenate operation.

    Args:
        graph:
        node (Node): current_node of the operation
        stacked_nodes (list(Node)): nodes to concatenate
        axis (int): axis to concatenate
    """
    def __init__(self, graph=None, node=None, stacked_nodes=None, axis=-1):
        self.graph = graph
        self.node = node
        self.stacked_nodes = stacked_nodes
        self.axis = axis

    def is_set(self):
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.graph.add_edge(n, self.node)

    def __call__(self, values, **kwargs):
        if len(values) > 1:
            out = keras.layers.Concatenate(axis=-1)(values)
        else:
            out = values[0]
        return out

class Dense(Operation):
    """Multi Layer Perceptron operation.

    Help you to create a perceptron with n layers, m units per layer and an activation function.

    Args:
        layers (int): number of layers.
        units (int): number of units per layer.
        activation: an activation function from tensorflow.
    """
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def __str__(self):
        return f'Dense_{self.units}_{self.activation.__name__}'

    def __call__(self, inputs, **kwargs):
        out = keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=tf.initializers.random_uniform())(inputs[0])
        return out


class Dropout(Operation):
    """Dropout operation.

    Help you to create a dropout operation.

    Args:
        rate (float): rate of deactivated inputs.
    """
    def __init__(self, rate):
        self.rate = rate

    def __str__(self):
        return f'Dropout({int((1.-self.rate)*100)})'

    def __call__(self, inputs, **kwargs):
        inpt = inputs[0]
        out = keras.layers.Dropout(rate=self.rate)(inpt)
        return out


dropout_ops = [Dropout(0.),
               Dropout(0.1),
               Dropout(0.2),
               Dropout(0.3),
               Dropout(0.4),
               Dropout(0.5),
               Dropout(0.6)]

class Identity(Operation):
    def __call__(self, inpt, **kwargs):
        return inpt[0]

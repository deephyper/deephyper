import collections

import tensorflow as tf

from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Zero, Connect, AddByProjecting, Identity

Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)


class DenseSkipCoSpace(KSearchSpace):
    def __init__(
        self,
        input_shape,
        output_shape,
        batch_size=None,
        seed=None,
        regression=True,
        num_layers=10,
        dropout=0.0,
    ):
        super().__init__(input_shape, output_shape, batch_size=batch_size, seed=seed)

        self.regression = regression
        self.num_layers = num_layers
        self.dropout = dropout

    def build(self):

        source = prev_input = self.input_nodes[0]

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(self.num_layers):
            vnode = VariableNode()
            self.add_dense_to_(vnode)

            self.connect(prev_input, vnode)

            # * Cell output
            cell_output = vnode

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(self, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self, anchor))
                self.connect(skipco, cmerge)

            prev_input = cmerge

            # ! for next iter
            anchor_points.append(prev_input)

        if self.dropout >= 0.0:
            dropout_node = ConstantNode(op=Dropout(rate=self.dropout))
            self.connect(prev_input, dropout_node)
            prev_input = dropout_node

        output_node = ConstantNode(
            Dense(
                self.output_shape[0], activation=None if self.regression else "softmax"
            )
        )
        self.connect(prev_input, output_node)

        return self

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case

        activations = [None, tf.nn.swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
        for units in range(16, 97, 16):
            for activation in activations:
                node.add_op(Dense(units=units, activation=activation))


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    shapes = dict(input_shape=(10,), output_shape=(1,))
    space = DenseSkipCoSpace(**shapes).build()
    model = space.sample()
    plot_model(model)

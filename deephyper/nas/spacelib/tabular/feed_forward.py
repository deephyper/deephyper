import tensorflow as tf

from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import Identity, operation

Dense = operation(tf.keras.layers.Dense)


class FeedForwardSpace(KSearchSpace):
    """Simple search space for a feed-forward neural network. No skip-connection. Looking over the number of units per layer and the number of layers.

    Args:
        input_shape (tuple, optional): True shape of inputs (no batch size dimension). Defaults to (2,).
        output_shape (tuple, optional): True shape of outputs (no batch size dimension).. Defaults to (1,).
        num_layers (int, optional): Maximum number of layers to have. Defaults to 10.
        num_units (tuple, optional): Range of number of units such as range(start, end, step_size). Defaults to (1, 11).
        regression (bool, optional): A boolean defining if the model is a regressor or a classifier. Defaults to True.
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        batch_size=None,
        seed=None,
        regression=True,
        num_units=(1, 11),
        num_layers=10,
    ):
        super().__init__(input_shape, output_shape, batch_size=batch_size, seed=seed)
        self.regression = regression
        self.num_units = num_units
        self.num_layers = num_layers

    def build(self):

        prev_node = self.input_nodes[0]

        for _ in range(self.num_layers):
            vnode = VariableNode()
            vnode.add_op(Identity())
            for i in range(*self.num_units):
                vnode.add_op(Dense(i, tf.nn.relu))

            self.connect(prev_node, vnode)
            prev_node = vnode

        output_node = ConstantNode(
            Dense(
                self.output_shape[0], activation=None if self.regression else "softmax"
            )
        )
        self.connect(prev_node, output_node)

        return self


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    shapes = dict(input_shape=(10,), output_shape=(1,))
    space = FeedForwardSpace(**shapes).build()
    model = space.sample()
    plot_model(model)

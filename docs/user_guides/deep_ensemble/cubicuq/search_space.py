import collections

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from deephyper.nas.space import KSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Identity
from deephyper.nas.space.op import operation

Dense = operation(tf.keras.layers.Dense)
DistributionLambda = operation(tfp.layers.DistributionLambda)

ACTIVATIONS = [
    tf.keras.activations.elu,
    tf.keras.activations.gelu,
    tf.keras.activations.hard_sigmoid,
    tf.keras.activations.linear,
    tf.keras.activations.relu,
    tf.keras.activations.selu,
    tf.keras.activations.sigmoid,
    tf.keras.activations.softplus,
    tf.keras.activations.softsign,
    tf.keras.activations.swish,
    tf.keras.activations.tanh,
]


class UQRegressionFactory(SpaceFactory):
    def build(
        self,
        input_shape,
        output_shape,
        num_layers=3,
        **kwargs,
    ):

        self.ss = KSearchSpace(input_shape, output_shape)
        output_dim = output_shape[0]
        source = self.ss.input_nodes[0]

        out_sub_graph = self.build_sub_graph(source, num_layers)

        node2 = ConstantNode(op=Dense(output_dim * 2))  # means and stddev
        self.ss.connect(out_sub_graph, node2)

        node3 = ConstantNode(
            op=DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :output_dim],
                    scale=1e-3 + tf.math.softplus(0.05 * t[..., output_dim:]),
                )
            )
        )
        self.ss.connect(node2, node3)

        return self.ss

    def build_sub_graph(self, input_, num_layers=3):
        source = prev_input = input_

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(num_layers):
            vnode = VariableNode()
            self.add_dense_to_(vnode)

            self.ss.connect(prev_input, vnode)

            # * Cell output
            cell_output = vnode

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(self.ss, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))
                self.ss.connect(skipco, cmerge)

            prev_input = cmerge

            # ! for next iter
            anchor_points.append(prev_input)

        return prev_input

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case
        for units in range(16, 16 * 16 + 1, 16):
            for activation in ACTIVATIONS:
                node.add_op(Dense(units=units, activation=activation))


def create_search_space(input_shape=(1,), output_shape=(1,), **kwargs):
    return UQRegressionFactory()(input_shape, output_shape, **kwargs)


if __name__ == "__main__":
    shapes = dict(input_shape=(1,), output_shape=(1,))
    factory = UQRegressionFactory()
    factory.plot_model(**shapes)
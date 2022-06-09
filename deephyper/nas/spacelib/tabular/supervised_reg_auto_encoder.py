import tensorflow as tf

from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import Identity, operation

Dense = operation(tf.keras.layers.Dense)


class SupervisedRegAutoEncoderSpace(KSearchSpace):
    def __init__(
        self,
        input_shape,
        output_shape,
        batch_size=None,
        seed=None,
        units=[128, 64, 32, 16, 8, 16, 32, 64, 128],
        num_layers=5,
    ):
        super().__init__(input_shape, output_shape, batch_size=batch_size, seed=seed)

        self.units = units
        self.num_layers = num_layers

    def build(self):

        inp = self.input_nodes[0]

        # auto-encoder
        units = [128, 64, 32, 16, 8, 16, 32, 64, 128]
        prev_node = inp
        d = 1
        for i in range(len(units)):
            vnode = VariableNode()
            vnode.add_op(Identity())
            if d == 1 and units[i] < units[i + 1]:
                d = -1
                for u in range(min(2, units[i]), max(2, units[i]) + 1, 2):
                    vnode.add_op(Dense(u, tf.nn.relu))
                latente_space = vnode
            else:
                for u in range(
                    min(units[i], units[i + d]), max(units[i], units[i + d]) + 1, 2
                ):
                    vnode.add_op(Dense(u, tf.nn.relu))
            self.connect(prev_node, vnode)
            prev_node = vnode

        out2 = ConstantNode(op=Dense(self.output_shape[0][0], name="output_0"))
        self.connect(prev_node, out2)

        # regressor
        prev_node = latente_space
        # prev_node = inp
        for _ in range(self.num_layers):
            vnode = VariableNode()
            for i in range(16, 129, 16):
                vnode.add_op(Dense(i, tf.nn.relu))

            self.connect(prev_node, vnode)
            prev_node = vnode

        out1 = ConstantNode(op=Dense(self.output_shape[1][0], name="output_1"))
        self.connect(prev_node, out1)

        return self


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    shapes = dict(input_shape=(100,), output_shape=[(100,), (10,)])
    space = SupervisedRegAutoEncoderSpace(**shapes).build()
    model = space.sample()
    plot_model(model)

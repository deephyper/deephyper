import tensorflow as tf

from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Concatenate

Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)


class OneLayerSpace(KSearchSpace):
    def __init__(
        self, input_shape, output_shape, batch_size=None, seed=None, regression=True
    ):
        super().__init__(input_shape, output_shape, batch_size=batch_size, seed=seed)
        self.regression = regression

    def build(self):

        if type(self.input_shape) is list:
            vnodes = []
            for i in range(len(self.input_shape)):
                vn = self.gen_vnode()
                vnodes.append(vn)
                self.connect(self.input_nodes[i], vn)
                print(i)

            prev_node = ConstantNode(Concatenate(self, vnodes))

        else:

            prev_node = self.gen_vnode()
            self.connect(self.input_nodes[0], prev_node)

        output_node = ConstantNode(
            Dense(
                self.output_shape[0], activation=None if self.regression else "softmax"
            )
        )
        self.connect(prev_node, output_node)

        return self

    def gen_vnode(self) -> VariableNode:
        vnode = VariableNode()
        for i in range(1, 1000):
            vnode.add_op(Dense(i, tf.nn.relu))
        return vnode


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    shapes = dict(input_shape=[(10,), (10,)], output_shape=(1,))
    space = OneLayerSpace(**shapes).build()
    model = space.sample()
    plot_model(model)

import pytest


@pytest.mark.nas
def test_basic_space(verbose=0):
    import tensorflow as tf

    from deephyper.nas import KSearchSpace
    from deephyper.nas.node import VariableNode, ConstantNode
    from deephyper.nas.operation import operation, Identity

    Dense = operation(tf.keras.layers.Dense)

    class BasicSpace(KSearchSpace):
        def __init__(self, input_shape, output_shape, batch_size=None, *args, **kwargs):
            super().__init__(
                input_shape, output_shape, batch_size=batch_size, *args, **kwargs
            )

        def build(self):

            input_node = self.input[0]

            dense = VariableNode()
            dense.add_op(Identity())
            for i in range(1, 1000):
                dense.add_op(Dense(i))
            self.connect(input_node, dense)

            output_node = ConstantNode(Dense(self.output_shape[0]))
            self.connect(dense, output_node)

    space = BasicSpace(input_shape=(1,), output_shape=(1,))
    space.build()

    model_1 = space.sample([1])

    if verbose:
        model_1.summary()

    model_2 = space.sample()

    if verbose:
        model_2.summary()

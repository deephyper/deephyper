import unittest

import pytest


@pytest.mark.nas
class TestKSearchSpace(unittest.TestCase):
    def test_create(self):
        import tensorflow as tf
        from deephyper.nas import KSearchSpace
        from deephyper.nas.node import VariableNode
        from deephyper.nas.operation import operation

        Dense = operation(tf.keras.layers.Dense)

        class TestSpace(KSearchSpace):
            def __init__(self, input_shape, output_shape):
                super().__init__(input_shape, output_shape)

            def build(self):
                vnode = VariableNode()

                self.connect(self.input_nodes[0], vnode)

                vnode.add_op(Dense(1))

                return self

        space = TestSpace((5,), (1,)).build()
        model = space.sample()

    def test_create_more_nodes(self):
        import tensorflow as tf
        from deephyper.nas import KSearchSpace
        from deephyper.nas.node import VariableNode
        from deephyper.nas.operation import operation

        Dense = operation(tf.keras.layers.Dense)

        class TestSpace(KSearchSpace):
            def __init__(self, input_shape, output_shape):
                super().__init__(input_shape, output_shape)

            def build(self):
                vnode1 = VariableNode()
                self.connect(self.input_nodes[0], vnode1)

                vnode1.add_op(Dense(10))

                vnode2 = VariableNode()
                vnode2.add_op(Dense(1))

                self.connect(vnode1, vnode2)

                return self

        space = TestSpace((5,), (1,)).build()
        model = space.sample()

    def test_create_multiple_inputs_with_one_vnode(self):
        import tensorflow as tf
        from deephyper.nas import KSearchSpace
        from deephyper.nas.node import ConstantNode, VariableNode
        from deephyper.nas.operation import operation, Concatenate

        Dense = operation(tf.keras.layers.Dense)

        class TestSpace(KSearchSpace):
            def __init__(self, input_shape, output_shape):
                super().__init__(input_shape, output_shape)

            def build(self):
                merge = ConstantNode()
                merge.set_op(Concatenate(self, self.input_nodes))

                vnode1 = VariableNode()
                self.connect(merge, vnode1)

                vnode1.add_op(Dense(1))

                return self

        space = TestSpace([(5,), (5,)], (1,)).build()
        model = space.sample()

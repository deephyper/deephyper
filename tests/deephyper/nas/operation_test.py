import unittest

import tensorflow as tf
from deephyper.nas import AutoKSearchSpace
from deephyper.nas.node import VariableNode, ConstantNode
from deephyper.nas.operation import operation
from deephyper.nas.operation import Connect


class TestOperation(unittest.TestCase):


    def test_create_search_space(self):
        input_shape, output_shape = (2,), (1,)

        Dense = operation(tf.keras.layers.Dense)

        ss = AutoKSearchSpace(input_shape, output_shape, regression=True)

        vnode1 = VariableNode()
        for _ in range(1, 11):
            vnode1.add_op(Dense(10))

        ss.connect(ss.input_nodes[0], vnode1)

        ss.set_ops([0])
        ss.create_model()


    def test_connect(self):
        input_shape, output_shape = (2,), (1,)

        Dense = operation(tf.keras.layers.Dense)

        ss = AutoKSearchSpace(input_shape, output_shape, regression=True)

        node = ConstantNode(Connect(ss, ss.input_nodes[0]))

        output_node = ConstantNode(Dense(1))
        ss.connect(node, output_node)

        ss.set_ops([])
        ss.create_model()

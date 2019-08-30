import tensorflow as tf

from ..space import AutoKSearchSpace
from ..space.node import VariableNode
from ..space.op.op1d import Dense


def create_search_space(input_shape=(2,), output_shape=(1,), **kwargs):
    struct = AutoKSearchSpace(input_shape, output_shape, regression=True)

    prev_node = struct.input_nodes[0]

    for _ in range(20):
        vnode = VariableNode()
        for i in range(1, 11):
            vnode.add_op(Dense(i, tf.nn.relu))

        struct.connect(prev_node, vnode)

    return struct
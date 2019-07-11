import tensorflow as tf

from deephyper.search.nas.model.space.struct import AutoOutputStructure
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.op1d import Dense


def create_structure(input_shape=(2,), output_shape=(1,), **kwargs):
    struct = AutoOutputStructure(input_shape, output_shape, regression=True)

    vnode1 = VariableNode()
    for i in range(1, 11):
        vnode1.add_op(Dense(i, tf.nn.relu))

    struct.connect(struct.input_nodes[0], vnode1)

    return struct
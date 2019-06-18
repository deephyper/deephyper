import tensorflow as tf

from deephyper.search.nas.model.space.struct import DirectStructure
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.op1d import Dense


def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    struct = DirectStructure(input_shape, output_shape)

    vnode1 = VariableNode()
    vnode1.add_op(Dense(10, tf.nn.relu))
    vnode1.add_op(Dense(15, tf.nn.relu))
    vnode1.add_op(Dense(20, tf.nn.relu))

    struct.add_node(vnode1)

    return struct

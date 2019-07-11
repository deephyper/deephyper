import tensorflow as tf
from deephyper.search.nas.model.space.struct import AutoOutputStructure
from deephyper.search.nas.model.space.node import VariableNode


def create_structure(input_shape=(10,), output_shape=(1,), **kwargs):
    struct = AutoOutputStructure(input_shape, output_shape, regression=True)

    vnode = VariableNode()
    for num_units in range(1, 11):
        vnode.add_op(tf.keras.layers.Dense(num_units, tf.nn.relu))

    struct.connect(struct.input_nodes[0], vnode)

    return struct
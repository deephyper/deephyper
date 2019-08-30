import tensorflow as tf

from deephyper.search.nas.model.space.architecture import AutoKArchitecture
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.op1d import Operation

def test_create_architecture(input_shape=(2,), output_shape=(1,), **kwargs):
    struct = AutoKArchitecture(input_shape, output_shape, regression=True)

    vnode1 = VariableNode()
    for _ in range(1, 11):
        vnode1.add_op(Operation(layer=tf.keras.layers.Dense(10)))

    struct.connect(struct.input_nodes[0], vnode1)

    struct.set_ops([0])
    struct.create_model()


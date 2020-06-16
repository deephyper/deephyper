import collections

import tensorflow as tf

from deephyper.search.nas.model.space import AutoKSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.op1d import Dense, Identity, Flatten
from deephyper.search.nas.model.space.op.gnn import EdgeConditionedConv2, GlobalAvgPool2


def add_gcn_to_(node):
    activations = [tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for channels in range(16, 49, 16):
        for activation in activations:
            node.add_op(EdgeConditionedConv2(channels=channels, activation=activation, use_bias=False))
    return


def create_search_space(input_shape=None,
                        output_shape=None,
                        num_layers=1,
                        *args, **kwargs):
    if output_shape is None:
        output_shape = (1,)
    if input_shape is None:
        input_shape = [(8, 4), (8, 8), (8, 8, 3)]
    arch = AutoKSearchSpace(input_shape, output_shape, regression=True)
    prev_input = arch.input_nodes[0]
    prev_input1 = arch.input_nodes[1]
    prev_input2 = arch.input_nodes[2]

    vnode = VariableNode()
    add_gcn_to_(vnode)
    arch.connect(prev_input, vnode)
    arch.connect(prev_input1, vnode)
    arch.connect(prev_input2, vnode)
    prev_input = vnode

    vnode2 = VariableNode()
    add_gcn_to_(vnode2)
    arch.connect(prev_input, vnode2)
    arch.connect(prev_input1, vnode2)
    arch.connect(prev_input2, vnode2)
    prev_input = vnode2

    pnode = VariableNode()
    pnode.add_op(GlobalAvgPool2())
    arch.connect(prev_input, pnode)
    prev_input = pnode

    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [prev_input], activation="relu"))

    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model

    search_space = create_search_space(num_layers=10)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file='sampled_neural_network.png', show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")

    from gnn_test2.qm9.load_data import load_data_qm9
    ([X_train, A_train, E_train], y_train), ([X_test, A_test, E_test], y_test) = load_data_qm9()
    model.compile(loss="mse", optimizer="adam")
    model.fit([X_train, A_train, E_train], y_train,
              validation_data=([X_test, A_test, E_test], y_test),
              epochs=20)


if __name__ == '__main__':
    test_create_search_space()
    print()

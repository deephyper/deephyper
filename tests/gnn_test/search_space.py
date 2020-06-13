import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import collections

import tensorflow as tf

from deephyper.search.nas.model.space import AutoKSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.gnn import GraphConv2, EdgeConditionedConv2


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
    # * Cell output
    cell_output = vnode

    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))
    prev_input = cmerge

    vnode = VariableNode()
    add_gcn_to_(vnode)
    arch.connect(prev_input, vnode)
    arch.connect(prev_input1, vnode)
    arch.connect(prev_input2, vnode)
    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))
    prev_input = cmerge

    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    search_space = create_search_space(num_layers=2)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file='sampled_neural_network.png', show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")
    return model


if __name__ == '__main__':
    test_create_search_space()

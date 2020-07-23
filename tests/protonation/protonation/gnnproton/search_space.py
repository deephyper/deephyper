import collections
import tensorflow as tf
from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.gnn import GlobalAvgPool, GlobalSumPool, GlobalMaxPool, SPARSE_MPNN, GlobalAttentionPool, GlobalAttentionSumPool, Apply1DMask
from deephyper.search.nas.model.space.op.op1d import Dense, Flatten
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.basic import Tensor


def add_gat_to_(node):
    """
    Function to add operations to graph convolution variable node
    Args:
        node: node object

    Returns:

    """
    # node.add_op(GraphIdentity())
    state_dims = [4, 8, 16, 32]
    Ts = [1, 2, 3, 4]
    attn_methods = ['const', 'gcn', 'gat', 'sym-gat', 'linear', 'gen-linear', 'cos']
    attn_heads = [1, 2, 4, 6]
    aggr_methods = ['max', 'mean', 'sum']
    update_methods = ['gru', 'mlp']
    activations = [tf.keras.activations.sigmoid,
                   tf.keras.activations.tanh,
                   tf.keras.activations.relu,
                   tf.keras.activations.linear,
                   tf.keras.activations.elu,
                   tf.keras.activations.softplus,
                   tf.nn.leaky_relu,
                   tf.nn.relu6]

    for state_dim in state_dims:
        for T in Ts:
            for attn_method in attn_methods:
                for attn_head in attn_heads:
                    for aggr_method in aggr_methods:
                        for update_method in update_methods:
                            for activation in activations:
                                node.add_op(SPARSE_MPNN(state_dim=state_dim,
                                                        T=T,
                                                        attn_method=attn_method,
                                                        attn_head=attn_head,
                                                        aggr_method=aggr_method,
                                                        update_method=update_method,
                                                        activation=activation))
    return


def add_global_pooling_to_(node):
    """
    Function to add operations to dense variable node
    Args:
        node: node object

    Returns:

    """
    for functions in [GlobalSumPool, GlobalMaxPool, GlobalAvgPool]:
        for axis in [-1, -2]:  # Pool in terms of nodes or features
            node.add_op(functions(axis=axis))
    node.add_op(Flatten())
    for state_dim in [16, 32, 64]:
        node.add_op(GlobalAttentionPool(state_dim=state_dim))
    node.add_op(GlobalAttentionSumPool())
    return


def create_search_space(input_shape=None,
                        output_shape=(33,),
                        num_gcn_layers=3,
                        num_dense_layers=2,
                        *args, **kwargs):
    """
    A function to create keras search sapce
    Args:
        input_shape: list of tuples
        output_shape: a tuple
        num_gcn_layers: int, number of graph convolution layers
        num_dense_layers: int, number of dense layers
        *args:
        **kwargs:

    Returns:

    """
    if input_shape is None:
        input_shape = [(32+1, 23),
                       (32+1+94+1, 2),
                       (32+1+94+1, 6),
                       (32+1, ),
                       (32+1+94+1, )]
    arch = KSearchSpace(input_shape, output_shape, regression=True)
    source = prev_input = arch.input_nodes[0]
    prev_input1 = arch.input_nodes[1]
    prev_input2 = arch.input_nodes[2]
    prev_input3 = arch.input_nodes[3]
    prev_input4 = arch.input_nodes[4]

    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=3)

    count_gcn_layers = 0
    count_dense_layers = 0
    for _ in range(num_gcn_layers):
        graph_attn_cell = VariableNode()
        add_gat_to_(graph_attn_cell)  #
        arch.connect(prev_input, graph_attn_cell)
        arch.connect(prev_input1, graph_attn_cell)
        arch.connect(prev_input2, graph_attn_cell)
        arch.connect(prev_input3, graph_attn_cell)
        arch.connect(prev_input4, graph_attn_cell)

        cell_output = graph_attn_cell
        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        prev_input = cmerge
        anchor_points.append(prev_input)
        count_gcn_layers += 1

    global_pooling_node = VariableNode()
    add_global_pooling_to_(global_pooling_node)
    arch.connect(prev_input, global_pooling_node)  # result from graph conv (?, 23, ?) --> Global pooling (?, 23)
    prev_input = global_pooling_node

    flatten_node = ConstantNode()
    flatten_node.set_op(Flatten())
    arch.connect(prev_input, flatten_node)  # result from graph conv (?, 23) --> Flatten
    prev_input = flatten_node

    for _ in range(num_dense_layers):
        dense_node = ConstantNode()
        dense_node.set_op(Dense(32, activation='relu'))
        arch.connect(prev_input, dense_node)
        prev_input = dense_node
        count_dense_layers += 1

    output_node = ConstantNode()
    output_node.set_op(Dense(output_shape[0], activation='linear'))
    arch.connect(prev_input, output_node)
    prev_input = output_node
    mask_node = ConstantNode()
    mask_node.set_op(Apply1DMask())
    arch.connect(prev_input, mask_node)
    arch.connect(prev_input3, mask_node)
    return arch


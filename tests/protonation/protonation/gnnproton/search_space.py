import collections
import tensorflow as tf
from deephyper.search.nas.model.space import AutoKSearchSpace, KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.gnn import EdgeConditionedConv2, GlobalMaxPool2, GraphIdentity, GlobalAvgPool2,\
    GlobalSumPool2, Apply1DMask
from deephyper.search.nas.model.space.op.op1d import Dense, Identity, Flatten
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.basic import Tensor


def add_gcn_to_(node):
    """
    Function to add operations to graph convolution variable node
    Args:
        node: node object

    Returns:

    """
    node.add_op(GraphIdentity())
    activations = [tf.nn.relu, tf.nn.tanh]

    for channels in [16, 32]:
        for activation in activations:
            node.add_op(EdgeConditionedConv2(channels=channels, activation=activation))
    return


# def add_global_pool_to_(node):
#     """
#     Function to add operations to graph global pooling variable node
#     Args:
#         node: node object
#
#     Returns:
#
#     """
#     node.add_op(GlobalMaxPool2())
#     node.add_op(GlobalAvgPool2())
#     node.add_op(GlobalSumPool2())
#     return


def add_dense_to_(node):
    """
    Function to add operations to dense variable node
    Args:
        node: node object

    Returns:

    """
    node.add_op(Identity())

    activations = [tf.nn.relu, tf.nn.tanh]
    for units in [16, 32]:
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))
    return


def create_search_space(input_shape=None,
                        output_shape=(31, ),
                        num_gcn_layers=2,
                        num_dense_layers=1,
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
        input_shape = [(31, 23), (31, 31), (31, 31, 6), (31,)]
    print(f"================ Create Search Space ================")
    arch = KSearchSpace(input_shape, output_shape, regression=True)
    source = prev_input = arch.input_nodes[0]
    prev_input1 = arch.input_nodes[1]
    prev_input2 = arch.input_nodes[2]
    prev_input3 = arch.input_nodes[3]

    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=2)

    count_gcn_layers = 0
    count_dense_layers = 0
    for _ in range(num_gcn_layers):
        vnode1 = VariableNode()
        add_gcn_to_(vnode1)
        arch.connect(prev_input, vnode1)
        arch.connect(prev_input1, vnode1)
        arch.connect(prev_input2, vnode1)

        cell_output = vnode1
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

    vnode2 = ConstantNode()
    vnode2.set_op(GlobalMaxPool2())
    arch.connect(prev_input, vnode2)

    cell_output = vnode2
    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

    source2 = prev_input = cmerge
    # look over skip connections within a range of the 3 previous nodes
    anchor_points2 = collections.deque([source2], maxlen=2)

    for _ in range(num_dense_layers):
        vnode3 = VariableNode()
        add_dense_to_(vnode3)

        arch.connect(prev_input, vnode3)

        cell_output = vnode3
        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

        for anchor in anchor_points2:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        prev_input = cmerge
        anchor_points2.append(prev_input)
        count_dense_layers += 1

    fnode = ConstantNode()
    fnode.set_op(Flatten())
    arch.connect(prev_input, fnode)
    prev_input = fnode

    onode = ConstantNode()
    onode.set_op(Dense(output_shape[0], activation='relu'))
    arch.connect(prev_input, onode)
    prev_input = onode

    mnode = ConstantNode()
    mnode.set_op(Apply1DMask())
    arch.connect(prev_input, mnode)
    arch.connect(prev_input3, mnode)

    print(f"================ Create Search Space Finished w/ {count_gcn_layers} GCN Layers, {count_dense_layers}"
          f" Dense Layers================")

    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model

    search_space = create_search_space(input_shape=[(31, 23), (31, 31), (31, 31, 6), (31, )],
                                       output_shape=(31, ),
                                       num_gcn_layers=2,
                                       num_dense_layers=1)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file='sampled_neural_network.png', show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")
    print(f"The size of search space is {search_space.size}")
    #
    # from protonation.gnnproton.load_data import load_data
    # from tensorflow.keras.callbacks import ModelCheckpoint
    # from tensorflow.keras.optimizers import Adam
    # ([X_train, A_train, E_train, m_train], y_train), ([X_valid, A_valid, E_valid, m_valid], y_valid) = load_data()
    # model.compile(loss="mse", optimizer=Adam(learning_rate=1e-3))
    # mcp = ModelCheckpoint("./1.h5", monitor="val_loss", mode="min", save_best_only=True,
    #                       save_weights_only=True, verbose=2)
    # model.fit([X_train, A_train, E_train, m_train], y_train,
    #          validation_data=([X_valid, A_valid, E_valid, m_valid], y_valid),
    #          epochs=100, callbacks=[mcp], verbose=0)
    # model.load_weights("./1.h5")
    # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    # from tabulate import tabulate
    # mse_train = mean_squared_error(y_train.ravel(), model.predict([X_train, A_train, E_train, m_train]).ravel())
    # mae_train = mean_absolute_error(y_train.ravel(), model.predict([X_train, A_train, E_train, m_train]).ravel())
    # r2_train = r2_score(y_train.ravel(), model.predict([X_train, A_train, E_train, m_train]).ravel())
    # mse_valid = mean_squared_error(y_valid.ravel(), model.predict([X_valid, A_valid, E_valid, m_valid]).ravel())
    # mae_valid = mean_absolute_error(y_valid.ravel(), model.predict([X_valid, A_valid, E_valid, m_valid]).ravel())
    # r2_valid = r2_score(y_valid.ravel(), model.predict([X_valid, A_valid, E_valid, m_valid]).ravel())
    # print(tabulate([['Train', mse_train, mae_train, r2_train], ['Valid', mse_train, mae_valid, r2_valid]],
    #                headers=['', 'MSE', 'MAE', 'R2']))


if __name__ == '__main__':
    test_create_search_space()

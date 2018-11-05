from deephyper.search.nas.cell import Node, SequentialStructure
from deephyper.search.nas.cell.cnn import (create_cnn_normal_cell,
                                           create_cnn_reduction_cell)
from deephyper.search.nas.operation.basic import Tensor


def create_structure(input_tensor, n_normal):
    """
    Create a structure corresponding to the nas-net search space :
    https://arxiv.org/abs/1707.07012

    Args:
        input_tensor (tensor): the input tensor of the structure.
        n_normal (int): number of normal cells to stack.

    Return: a SequentialStructure object corresponding to the google nas net architecture idea, NxNormal -> Reduction -> NxNormal -> Reduction -> NxNormal.
    """
    # Input Node
    input_node = Node('Input')
    input_node.add_op(Tensor(input_tensor))
    input_node.set_op(0)
    inputs = [input_node]

    # Creation of the structure
    net_struct = SequentialStructure(inputs, merge_mode='concat_2d')

    # N x Normal Cells
    net_struct.add_cell_f(lambda : create_cnn_normal_cell(inputs), 2)
    for n in range(n_normal-1):
        net_struct.add_cell_f(create_cnn_normal_cell, 2)

    # Reduction Cell
    net_struct.add_cell_f(create_cnn_reduction_cell, 2)

    # N x Normal Cells
    net_struct.add_cell_f(create_cnn_normal_cell, 1)
    for n in range(n_normal-1):
        net_struct.add_cell_f(create_cnn_normal_cell, 2)

    # Reduction Cell
    net_struct.add_cell_f(create_cnn_reduction_cell, 2)

    # N x Normal Cells
    net_struct.add_cell_f(create_cnn_normal_cell, 1)
    for n in range(n_normal-1):
        net_struct.add_cell_f(create_cnn_normal_cell, 2)

    return net_struct

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from random import random
    import time


    inpt = tf.constant(np.zeros((2, 28, 28, 3)), dtype=tf.float32)
    n_normal = 2
    net_struct = create_structure(inpt, n_normal)

    ops = [random() for _ in range(net_struct.num_nodes)]
    normal_cell_ops = [random() for _ in range(net_struct.num_nodes_cell(i=0)[0])]
    reduct_cell_ops = [random() for _ in range(net_struct.num_nodes_cell(i=0)[0])]
    ops = normal_cell_ops*n_normal + reduct_cell_ops + normal_cell_ops*n_normal + reduct_cell_ops + normal_cell_ops*n_normal
    print(f'ops: {ops}')
    net_struct.set_ops(ops)
    net_struct.draw_graphviz('google_nas_net.dot')

    t_s = time.time()
    out = net_struct.create_tensor()
    t_e = time.time()
    dur = t_e - t_s

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run(out)

        print(f'res: {res}')
        print(f'input shape: {inpt.get_shape()}')
        print(f'outpt shape: {np.shape(res)}')
        print(f'num_nodes: {net_struct.num_nodes}')
        print(f'dur create_tensor: {dur}')

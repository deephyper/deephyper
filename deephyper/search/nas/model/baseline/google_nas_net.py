from deephyper.search.nas.cell import Node, SequentialStructure
from deephyper.search.nas.cell.cnn import (create_cnn_normal_cell,
                                           create_cnn_reduction_cell)
from deephyper.search.nas.operation.basic import Tensor
from deephyper.search.nas.cell import Block, Cell, Node
from deephyper.search.nas.operation.basic import Add, Concat, Connect
from deephyper.search.nas.operation.cnn import (AvgPooling2D, Convolution2D,
                                                DepthwiseSeparable2D,
                                                Dilation2D, IdentityConv2D,
                                                MaxPooling2D)

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


def create_block(cell, input_nodes, stride):
    n_ha = Node('Hidden')
    for inpt in input_nodes:
        n_ha.add_op(Connect(cell.graph, inpt, n_ha))
    n_hb = Node('Hidden')
    for inpt in input_nodes:
        n_hb.add_op(Connect(cell.graph, inpt, n_hb))


    identity = IdentityConv2D(stride=stride)

    conv_1x7 = Convolution2D(1, 7, stride=stride)
    conv_7x1 = Convolution2D(7, 1, stride=stride)
    conv_1x1 = Convolution2D(1, 1, stride=stride)
    conv_1x3 = Convolution2D(1, 3, stride=stride)
    conv_3x1 = Convolution2D(3, 1, stride=stride)
    conv_3x3 = Convolution2D(3, 3, stride=stride)

    maxpool_5x5 = MaxPooling2D(5, 5, stride=stride)
    maxpool_3x3 = MaxPooling2D(3, 3, stride=stride)
    maxpool_7x7 = MaxPooling2D(7, 7, stride=stride)

    avgpool_3x3 = AvgPooling2D(3, 3, stride=stride)

    depsepconv_3x3 = DepthwiseSeparable2D(3, 3, stride=stride)
    depsepconv_5x5 = DepthwiseSeparable2D(5, 5, stride=stride)
    depsepconv_7x7 = DepthwiseSeparable2D(7, 7, stride=stride)

    dilation_3x3 = Dilation2D(3, 3, stride=stride)

    # identity, 1x7 then 7x1 convolution, 3x3 average pooling, 5x5 max pooling,
    # 1x1 convolution, 3x3 depthwise-separable conv, 7x7 depthwise-separable conv,
    # 1x3 then 3x1 convolution, 3x3 dilated convolution, 3x3 max pooling, 7x7 max pooling,
    # 3x3 convolution, 5x5 depthwise-separable conv
    cnn_ops = [
        identity,
        conv_1x7,
        conv_7x1,
        avgpool_3x3,
        maxpool_5x5,
        conv_1x1,
        depsepconv_3x3,
        depsepconv_7x7,
        conv_1x3,
        conv_3x1,
        dilation_3x3,
        maxpool_3x3,
        maxpool_7x7,
        conv_3x3,
        depsepconv_5x5]

    n_opa = Node('Op')
    for op in cnn_ops:
        n_opa.add_op(op)
    n_opb = Node('Op')
    for op in cnn_ops:
        n_opb.add_op(op)

    # (1) element-wise addition between two hidden states
    # (2) concatenation between two hidden states along the filter dimension
    n_combine = Node('Combine')
    n_combine.add_op(Add())
    n_combine.add_op(Concat(axis=3))

    block = Block()
    block.add_node(n_ha)
    block.add_node(n_hb)
    block.add_node(n_opa)
    block.add_node(n_opb)
    block.add_node(n_combine)

    block.add_edge(n_ha, n_opa)
    block.add_edge(n_hb, n_opb)
    block.add_edge(n_opa, n_combine)
    block.add_edge(n_opb, n_combine)

    return block

def create_cnn_base_cell(input_nodes, stride):
    """Create CNN base cell.

    Base cell correspond to the generic cell structure with a generic stride explained in the paper 'Learning Transferable Architectures for Scalable Image Recognition'.

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.
        stride (int): stride of all convolution opperations inside the cell.

    Returns:
        Cell: a Cell instance corresponding to the previous description.
    """
    cell = Cell(input_nodes)

    num_blocks = 5
    cell.add_block(create_block(cell, input_nodes, stride=stride))
    for _ in range(num_blocks-1):
        cell.add_block(create_block(cell, input_nodes, stride=stride))

    cell.set_outputs('concat', axis=3)
    return cell

def create_cnn_normal_cell(input_nodes):
    """Create CNN normal cell.

    Normal cell corresponds to Normal Cell of the paper 'Learning Transferable Architectures for Scalable Image Recognition' : https://arxiv.org/abs/1707.07012 .
    It corresponds to base cell with ``(stride == 1)``.

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance corresponding to the previous description.
    """
    return create_cnn_base_cell(input_nodes, 1)

def create_cnn_reduction_cell(input_nodes):
    """Create CNN normal cell.

    Normal cell corresponds to Normal Cell of the paper 'Learning Transferable Architectures for Scalable Image Recognition' : https://arxiv.org/abs/1707.07012 .
    It corresponds to base cell with ``(stride == 2)``.

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance corresponding to the previous description.
    """
    return create_cnn_base_cell(input_nodes, 2)



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

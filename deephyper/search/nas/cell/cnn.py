from deephyper.search.nas.cell import Block, Cell, Node
from deephyper.search.nas.operation.basic import Add, Concat, Connect
from deephyper.search.nas.operation.cnn import (AvgPooling2D, Convolution2D,
                                                DepthwiseSeparable2D,
                                                Dilation2D, IdentityConv2D,
                                                MaxPooling2D)


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

import tensorflow as tf

from deephyper.search.nas.model.baseline.util.struct import (create_seq_struct,
                                                             create_struct_full_skipco)
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Conv1D, Dense, Identity,
                                                      MaxPooling1D,
                                                      dropout_ops)


def create_cell_1(input_nodes):
    """Create a cell with convolution.

    Args:
        input_nodes (list(Node)): a list of input_nodes for this cell.

    Returns:
        Cell: the corresponding cell.
    """
    cell = Cell(input_nodes)

    def create_conv_block(input_nodes):
        # first node of block
        n1 = VariableNode('N1')
        for inpt in input_nodes:
            n1.add_op(Connect(cell.graph, inpt, n1))

        def create_conv_node(name):
            n = VariableNode(name)
            n.add_op(Identity())
            n.add_op(Conv1D(filter_size=5, num_filters=2))
            n.add_op(Conv1D(filter_size=5, num_filters=3))
            n.add_op(MaxPooling1D(pool_size=3, padding='same'))
            n.add_op(MaxPooling1D(pool_size=5, padding='same'))
            return n
        # second node of block
        n2 = create_conv_node('N2')

        n3 = create_conv_node('N3')

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block

    block1 = create_conv_block(input_nodes)
    block2 = create_conv_block(input_nodes)
    block3 = create_conv_block(input_nodes)

    cell.add_block(block1)
    cell.add_block(block2)
    cell.add_block(block3)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    return create_struct_full_skipco(
        input_shape,
        output_shape,
        create_cell_1,
        num_cells)

import tensorflow as tf

from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Dense, Identity,
                                                      dropout_ops)
from deephyper.search.nas.model.baseline.util.struct import create_struct_full_skipco


def create_dense_cell(input_nodes):
    """MLP type 2

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance.
    """
    cell = Cell(input_nodes)

    node = VariableNode(name='N')
    node.add_op(Dense(10, tf.nn.relu))
    node.add_op(Dense(10, tf.nn.relu))
    node.add_op(Dense(10, tf.nn.relu))
    node.add_op(Dense(10, tf.nn.relu))
    cell.graph.add_edge(input_nodes[0], node)

    # Block
    block = Block()
    block.add_node(node)

    cell.add_block(block)

    cell.set_outputs()
    return cell


def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    return create_struct_full_skipco(
        input_shape,
        output_shape,
        create_dense_cell,
        num_cells)

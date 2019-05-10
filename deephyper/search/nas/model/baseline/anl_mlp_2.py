import tensorflow as tf

from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Dense, Identity,
                                                      dropout_ops)
from deephyper.search.nas.model.baseline.util.struct import create_struct_full_skipco


def create_dense_cell_type2(input_nodes):
    """MLP type 2

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance.
    """
    cell = Cell(input_nodes)

    # first node of block
    n1 = VariableNode('N_0')
    for inpt in input_nodes:
        n1.add_op(Connect(cell.graph, inpt, n1))

    # second node of block
    mlp_op_list = list()
    mlp_op_list.append(Identity())
    mlp_op_list.append(Dense(5, tf.nn.relu))
    mlp_op_list.append(Dense(10, tf.nn.relu))
    mlp_op_list.append(Dense(20, tf.nn.relu))
    mlp_op_list.append(Dense(40, tf.nn.relu))
    mlp_op_list.append(Dense(80, tf.nn.relu))
    mlp_op_list.append(Dense(160, tf.nn.relu))
    mlp_op_list.append(Dense(320, tf.nn.relu))
    n2 = VariableNode('N_1')
    for op in mlp_op_list:
        n2.add_op(op)

    # third
    n3 = VariableNode('N_2')
    drop_ops = []
    drop_ops.extend(dropout_ops)
    for op in drop_ops:
        n3.add_op(op)

    # 1 Blocks
    block1 = Block()
    block1.add_node(n1)
    block1.add_node(n2)
    block1.add_node(n3)

    block1.add_edge(n1, n2)
    block1.add_edge(n2, n3)

    cell.add_block(block1)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    return create_struct_full_skipco(
        input_shape,
        output_shape,
        create_dense_cell_type2,
        num_cells)

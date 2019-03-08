import tensorflow as tf

from deephyper.search.nas.model.baseline.util.struct import create_seq_struct
from deephyper.search.nas.model.baseline.util.struct import create_struct_full_skipco
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.node import VariableNode
from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import (Dense, Identity,
                                                      dropout_ops)


def create_dense_cell_type1(input_nodes):
    """Dense type 1

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance.
    """
    cell = Cell(input_nodes)

    def create_block():
        # first node of block
        n1 = VariableNode('N1')
        for inpt in input_nodes:
            n1.add_op(Connect(cell.graph, inpt, n1))

        # second node of block
        mlp_op_list = list()
        mlp_op_list.append(Identity())
        mlp_op_list.append(Dense(5, tf.nn.relu))
        mlp_op_list.append(Dense(5, tf.nn.tanh))
        mlp_op_list.append(Dense(10, tf.nn.relu))
        mlp_op_list.append(Dense(10, tf.nn.tanh))
        mlp_op_list.append(Dense(20, tf.nn.relu))
        mlp_op_list.append(Dense(20, tf.nn.tanh))
        n2 = VariableNode('N2')
        for op in mlp_op_list:
            n2.add_op(op)

        # third node of block
        n3 = VariableNode('N3')
        for op in dropout_ops:
            n3.add_op(op)

        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        return block

    # 2 Blocks per cell
    block1 = create_block()
    block2 = create_block()

    cell.add_block(block1)
    cell.add_block(block2)

    cell.set_outputs()
    return cell

def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    # return create_seq_struct(
    return create_struct_full_skipco(
        input_shape,
        output_shape,
        create_dense_cell_type1,
        num_cells)

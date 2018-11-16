import tensorflow as tf

from deephyper.search.nas.model.space.op.basic import Connect
from deephyper.search.nas.model.space.op.op1d import Dense, dropout_ops, Identity

from deephyper.search.nas.model.space.node import Node
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.cell import Cell

from deephyper.search.nas.cell.structure import create_seq_structure

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
        n1 = Node('N1')
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
        n2 = Node('N2')
        for op in mlp_op_list:
            n2.add_op(op)

        # third node of block
        n3 = Node('N3')
        for op in mlp_op_list:
            n3.add_op(op)

        # fourth node of block
        n4 = Node('N4')
        for op in mlp_op_list:
            n4.add_op(op)

        # fifth
        n5 = Node('N5')
        for op in dropout_ops:
            n5.add_op(op)

        # 5 Blocks
        block = Block()
        block.add_node(n1)
        block.add_node(n2)
        block.add_node(n3)
        block.add_node(n4)
        block.add_node(n5)

        block.add_edge(n1, n2)
        block.add_edge(n2, n3)
        block.add_edge(n3, n4)
        block.add_edge(n4, n5)
        return block

    block1 = create_block()
    block2 = create_block()
    block3 = create_block()
    block4 = create_block()
    block5 = create_block()

    cell.add_block(block1)
    cell.add_block(block2)
    cell.add_block(block3)
    cell.add_block(block4)
    cell.add_block(block5)

    cell.set_outputs()
    return cell

def create_structure(input_tensor, num_cells):
    return create_seq_structure(input_tensor, create_dense_cell_type1, num_cells)

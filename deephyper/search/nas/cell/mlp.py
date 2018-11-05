import tensorflow as tf

from deephyper.search.nas.cell import Block, Cell, Node
from deephyper.search.nas.operation.basic import Connect, Identity
from deephyper.search.nas.operation.mlp import MLP, dropout_ops, mlp_ops


def create_dense_cell_example(input_nodes):
    """Example MLP cell.

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance.
    """
    cell = Cell(input_nodes)

    def create_block():
        n1 = Node('N1')
        for inpt in input_nodes:
            n1.add_op(Connect(cell.graph, inpt, n1))

        n2 = Node('N2')
        for op in mlp_ops:
            n2.add_op(op)


        block = Block()
        block.add_node(n1)
        block.add_node(n2)

        block.add_edge(n1, n2)
        return block

    block1 = create_block()
    block2 = create_block()

    cell.add_block(block1)
    cell.add_block(block2)
    cell.set_outputs('stack', axis=1)
    return cell

def create_dense_cell_type1(input_nodes):
    """MLP type 1

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
        mlp_op_list.append(MLP(1, 5, tf.nn.relu))
        mlp_op_list.append(MLP(1, 5, tf.nn.tanh))
        mlp_op_list.append(MLP(1, 10, tf.nn.relu))
        mlp_op_list.append(MLP(1, 10, tf.nn.tanh))
        mlp_op_list.append(MLP(1, 20, tf.nn.relu))
        mlp_op_list.append(MLP(1, 20, tf.nn.tanh))
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

    cell.set_outputs('stack', axis=1)
    return cell

def create_dense_cell_type2(input_nodes):
    """MLP type 2

    Args:
        input_nodes (list(Node)): possible inputs of the current cell.

    Returns:
        Cell: a Cell instance.
    """
    cell = Cell(input_nodes)

    # first node of block
    n1 = Node('N1')
    for inpt in input_nodes:
        n1.add_op(Connect(cell.graph, inpt, n1))

    # second node of block
    mlp_op_list = list()
    mlp_op_list.append(Identity())
    mlp_op_list.append(MLP(1, 5, tf.nn.relu))
    mlp_op_list.append(MLP(1, 10, tf.nn.relu))
    mlp_op_list.append(MLP(1, 20, tf.nn.relu))
    mlp_op_list.append(MLP(1, 40, tf.nn.relu))
    mlp_op_list.append(MLP(1, 80, tf.nn.relu))
    mlp_op_list.append(MLP(1, 160, tf.nn.relu))
    mlp_op_list.append(MLP(1, 320, tf.nn.relu))
    n2 = Node('N2')
    for op in mlp_op_list:
        n2.add_op(op)

    # third
    n3 = Node('N3')
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

    cell.set_outputs('stack', axis=1)
    return cell

def create_dense_cell_toy(input_nodes):
    cell = Cell(input_nodes)

    n1 = Node('N1')
    n1.add_op(Connect(cell.graph, input_nodes[-1], n1))

    # first node of block
    mlp_op_list = list()
    mlp_op_list.append(MLP(1, 5, tf.nn.relu))
    mlp_op_list.append(MLP(1, 10, tf.nn.relu))
    mlp_op_list.append(MLP(1, 20, tf.nn.relu))
    mlp_op_list.append(MLP(1, 40, tf.nn.relu))
    mlp_op_list.append(MLP(1, 80, tf.nn.relu))
    mlp_op_list.append(MLP(1, 160, tf.nn.relu))
    mlp_op_list.append(MLP(1, 320, tf.nn.relu))
    n2 = Node('N2')
    for op in mlp_op_list:
        n2.add_op(op)

    # 1 Blocks
    block1 = Block()
    block1.add_node(n1)
    block1.add_node(n2)
    block1.add_edge(n1, n2)

    cell.add_block(block1)

    cell.set_outputs('stack', axis=1)
    return cell

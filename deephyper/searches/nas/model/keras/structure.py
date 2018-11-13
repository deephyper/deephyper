import networkx as nx
from tensorflow import keras

from deephyper.searches.nas.model.keras.cell import Cell
from deephyper.searches.nas.cell.block import Block
from deephyper.searches.nas.cell.node import Node
from deephyper.searches.nas.cell.block import create_tensor_aux
from deephyper.searches.nas.operation.basic import (Connect, Tensor)
from deephyper.searches.nas.operation.keras import Concatenate


class Structure:
    graph = None
    inputs = None
    struct = None

    def draw_graphviz(self, path):
        with open(path, 'w') as f:
            try:
                nx.nx_agraph.write_dot(self.graph, f)
            except:
                pass

class KerasStructure(Structure):
    def __init__(self, input_shape, output_shape):
        """
        Create a new Sequential Structure.

        Args:
            inputs (tuple):
        """
        self.graph = nx.DiGraph()

        self.input_node = Node('Input', [])
        self.input_node.add_op(Tensor(keras.layers.Input(input_shape)))
        self.input_node.set_op(0)

        self.output_shape = output_shape
        self.output = None

        self.struct = []

    def __len__(self):
        return len(self.struct)

    def __getitem__(self, sliced):
        return self.struct[sliced]

    def append(self, cell):
        """
        Append a cell to the structure.
        """
        self.struct.append(cell)

    @property
    def max_num_ops(self):
        """
        Return the maximum number of operations of nodes of blocks of cells of struct.
        """
        mx = 0
        for c in self.struct:
            mx = max(mx, c.max_num_ops())
        return mx

    @property
    def num_nodes(self):
        """
        Return: the number of nodes which correspond to the number of operations to set.
        """
        n = 0
        for c in self.struct:
            n += c.num_nodes()
        return n

    def num_nodes_cell(self, i=None):
        """
        Args:
            i (int): index of cell i in self.struct

        Return: the number of nodes in cell i or the list of number of nodes for each cell
        """
        if i != None:
            return [self.struct[i].num_nodes()]
        else:
            return [self.struct[i].num_nodes() for i in range(len(self.struct))]

    @property
    def num_cells(self):
        """
        Return: the number of cell in the Structure.
        """
        return len(self.struct)

    def add_cell_f(self, func, num=1):
        """
        Add a new cell in the structure.

        Args:
            func (function): a function that return a cell with one argument list of input nodes.
            num (int): number of hidden state with which the new cell can connect or None means all previous hidden states
        """
        possible_inputs = [self.input_node]
        possible_inputs.extend([c.output for c in self.struct])
        if len(self.struct) > 0:
            if num is None:
                cell = func(possible_inputs[:])
            else:
                cell = func(possible_inputs[len(possible_inputs)-num:])
        else:
            cell = func()
        self.struct.append(cell)

    def set_ops(self, indexes):
        """
        Set the operations for each node of each cell of the structure.

        Args:
            indexes (list): element of list can be float in [0, 1] or int.
        """
        cursor = 0
        for c in self.struct:
            num_nodes = c.num_nodes()
            c.set_ops(indexes[cursor:cursor+num_nodes])
            cursor += num_nodes

            self.graph.add_nodes_from(c.graph.nodes())
            self.graph.add_edges_from(c.graph.edges())

        output_nodes = get_output_nodes(self.graph)
        node = Node('Structure_Output')
        node.add_op(Concatenate(self.graph, node, output_nodes))
        node.set_op(0)
        self.output = node

    def create_model(self, train):
        """
        Create the tensors corresponding to the structure.

        Args:
            train (bool): True if the network is built for training, False if the network is built for validation/testing (for example False will deactivate Dropout).

        Return:
            The output tensor.
        """

        output_tensor = create_tensor_aux(self.graph, self.output, train=train)
        output_tensor = keras.layers.Dense(self.output_shape[0])(output_tensor)
        input_tensor = self.input_node._tensor
        return keras.Model(inputs=input_tensor, outputs=output_tensor)

def get_output_nodes(graph):
    """
    Args:
        graph: (nx.Digraph)

    Return: the nodes without successors of a DiGraph.
    """
    nodes = list(graph.nodes())
    output_nodes = []
    for n in nodes:
        if len(list(graph.successors(n))) == 0:
            output_nodes.append(n)
    return output_nodes


def create_seq_struct_full_skipco(input_shape, output_shape, create_cell, num_cells):
    """
        Create a SequentialStructure object.

        Args:
            input_tensor (tensor): a tensorflow tensor object
            create_cell (function): function that create a cell, take one argument (inputs: list(None))
            num_cells (int): number of cells in the sequential structure

        Return: SequentialStructure object.
    """

    network = KerasStructure(input_shape, output_shape)
    input_node = network.input_node

    func = lambda: create_cell([input_node])
    network.add_cell_f(func)

    func = lambda x: create_cell(x)
    for i in range(num_cells-1):
        network.add_cell_f(func, num=None)

    return network

def create_dense_cell_type2(input_nodes):
    import tensorflow as tf
    from deephyper.searches.nas.operation.basic import Connect
    from deephyper.searches.nas.operation.keras import Dense, Identity, dropout_ops
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
    mlp_op_list.append(Dense(5, tf.nn.relu))
    mlp_op_list.append(Dense(10, tf.nn.relu))
    mlp_op_list.append(Dense(20, tf.nn.relu))
    mlp_op_list.append(Dense(40, tf.nn.relu))
    mlp_op_list.append(Dense(80, tf.nn.relu))
    mlp_op_list.append(Dense(160, tf.nn.relu))
    mlp_op_list.append(Dense(320, tf.nn.relu))
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

    cell.set_outputs()
    return cell

def test_keras_structure():
    import tensorflow as tf
    import numpy as np
    from random import random

    input_tensor = tf.keras.Input(shape=(3,))

    structure = create_seq_struct_full_skipco(input_tensor, create_dense_cell_type2, 5)
    ops = [random() for _ in range(structure.num_nodes)]
    print(f'ops: {ops}')
    print(f'num ops: {len(ops)}')

    structure.set_ops(ops)
    structure.draw_graphviz('test_keras.dot')
    model = structure.create_model(train=True)

    data = np.random.random((1000, 3))
    result = model.predict(data, batch_size=32)
    print(result)



if __name__ == '__main__':
    test_keras_structure()

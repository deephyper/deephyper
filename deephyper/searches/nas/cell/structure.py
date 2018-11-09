import networkx as nx

from deephyper.searches.nas.cell.cell import Cell
from deephyper.searches.nas.cell.block import Block
from deephyper.searches.nas.cell.node import Node
from deephyper.searches.nas.cell.block import create_tensor_aux
from deephyper.searches.nas.operation.basic import (Add, Concat, Connect,
                                                  Constant, Incr, Merge,
                                                  Tensor)


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

class SequentialStructure(Structure):
    def __init__(self, inputs, merge_mode='concat_flat'):
        """
        Create a new Sequential Structure.

        Args:
            inputs (list(Node)):
            merge_mode (str): 'concat_flat' or 'concat_2d'
        """
        self.graph = nx.DiGraph()
        self.inputs = inputs
        self.output = None
        self.struct = []
        self.merge_mode = merge_mode

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
        possible_inputs = [self.inputs[0]]
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
        merge_mode = self.merge_mode
        cursor = 0
        for c in self.struct:
            num_nodes = c.num_nodes()
            c.set_ops(indexes[cursor:cursor+num_nodes])
            cursor += num_nodes

            self.graph.add_nodes_from(c.graph.nodes())
            self.graph.add_edges_from(c.graph.edges())

        output_nodes = get_output_nodes(self.graph)
        node = Node('Structure_Output')
        if merge_mode == 'concat_flat':
            node.add_op(Merge(self.graph, node, output_nodes, axis=1))
        elif merge_mode == 'concat_2d':
            node.add_op(Concat(self.graph, node, output_nodes, axis=3, last=True))
        node.set_op(0)
        self.output = node

    def create_tensor(self, train):
        """
        Create the tensors corresponding to the structure.

        Args:
            train (bool): True if the network is built for training, False if the network is built for validation/testing (for example False will deactivate Dropout).

        Return:
            The output tensor.
        """
        return create_tensor_aux(self.graph, self.output, train=train)

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


def test_sequential_structure():
    import tensorflow as tf
    from nas.cell.cell import create_test_cell

    input_node = Node('Input')
    input_node.add_op(Constant(1))
    input_node.set_op(0)
    inputs = [input_node]

    network = SequentialStructure(inputs)

    func = lambda: create_test_cell(inputs)
    network.add_cell_f(func)

    func = lambda x: create_test_cell(x)
    network.add_cell_f(func)

    network.set_ops([0 for _ in range(network.num_nodes)])

    outputs = network.create_tensor()

    with tf.Session() as sess:
        res = sess.run(outputs)
        print(f'res: {res}')

def create_sequential_structure(input_tensor, create_cell, num_cells):
    """
        Create a SequentialStructure object.

        Args:
            input_tensor (tensor): a tensorflow tensor object
            create_cell (function): function that create a cell, take one argument (inputs: list(None))
            num_cells (int): number of cells in the sequential structure

        Return: SequentialStructure object.
    """
    input_node = Node('Input')
    input_node.add_op(Tensor(input_tensor))
    input_node.set_op(0)
    inputs = [input_node]

    network = SequentialStructure(inputs)

    func = lambda: create_cell(inputs)
    network.add_cell_f(func)

    func = lambda x: create_cell(x)
    for i in range(num_cells-1):
        network.add_cell_f(func, num=2)

    return network

def create_seq_struct_full_skipco(input_tensor, create_cell, num_cells):
    """
        Create a SequentialStructure object.

        Args:
            input_tensor (tensor): a tensorflow tensor object
            create_cell (function): function that create a cell, take one argument (inputs: list(None))
            num_cells (int): number of cells in the sequential structure

        Return: SequentialStructure object.
    """
    input_node = Node('Input')
    input_node.add_op(Tensor(input_tensor))
    input_node.set_op(0)
    inputs = [input_node]

    network = SequentialStructure(inputs)

    func = lambda: create_cell(inputs)
    network.add_cell_f(func)

    func = lambda x: create_cell(x)
    for i in range(num_cells-1):
        network.add_cell_f(func, num=None)

    return network

def test_sequential_structure_mlp():
    import os
    import tensorflow as tf
    import random
    from nas.cell.cell import create_dense_cell

    input_node = Node('Input')
    input_node.add_op(Constant([[1., 1., 1.], [2., 2., 2.]]))
    input_node.set_op(0)
    inputs = [input_node]

    network = SequentialStructure(inputs)

    func = lambda: create_dense_cell(inputs)
    network.add_cell_f(func)

    func = lambda x: create_dense_cell(x)
    for i in range(1):
        network.add_cell_f(func, num=1)


    num_nodes = network.num_nodes_cell(0)[0]
    network.set_ops([random.random() for _ in range(num_nodes)]*network.num_cells)

    here = os.path.dirname(os.path.abspath(__file__))
    network.draw_graphviz(here+'/test.dot')

    outputs = network.create_tensor()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        res = sess.run(outputs)
        print(f'res: {res}')

if __name__ == '__main__':
    # test_sequential_structure()
    test_sequential_structure_mlp()

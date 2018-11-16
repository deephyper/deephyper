import networkx as nx

from deephyper.search.nas.model.space.node import Node
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.op.keras import Concatenate


class Cell:
    """Create a new Cell object.

    Args:
        inputs (list(Node)): possible inputs of the cell
    """
    num = 0

    def __init__(self, inputs=None):
        Cell.num += 1
        self.num = Cell.num
        self.inputs = inputs if not inputs is None else []
        self.output = None
        self.blocks = []
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(inputs)

    @property
    def num_nodes(self):
        """
        Return:
            int the number of nodes of this cell.
        """
        n = 0
        for b in self.blocks:
            n += b.num_nodes()
        return n

    @property
    def action_nodes(self):
        l = []
        for b in self.blocks:
            l.extend(b.action_nodes)
        return l

    def set_outputs(self):
        '''
            Set the output rule for the cell.
            Args:
                - kind: 'stack', 'concat'
        '''
        output_node = Node(f'Cell_{self.num}_Output')
        stacked_nodes = []
        for b in self.blocks:
            stacked_nodes.extend(b.outputs)
        op = Concatenate(self.graph, output_node, stacked_nodes)

        output_node.add_op(op)
        output_node.set_op(0)
        self.output = output_node

    def set_inputs(self, inputs):
        '''
        Remove the previous inputs from the graph if set then add the new inputs.

        Args:
            inputs list(Node): possible inputs of the cell
        '''
        self.graph.remove_nodes_from(self.inputs)
        self.inputs = inputs
        self.graph.add_nodes_from(inputs)

    def add_block(self, block):
        '''
            Add a new Block object to the Cell.
            Args:
                - block: Block the new block to add to the Cell
        '''
        assert not block in self.blocks
        self.blocks.append(block)

    def max_num_ops(self):
        '''
            Return the maximum number of operations of nodes of blocks of cell.
        '''
        mx = 0
        for b in self.blocks:
            mx = max(mx, b.max_num_ops())
        return mx

    def add_edge(self, node1, node2):
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)

        self.graph.add_edge(node1, node2)

        if not(nx.is_directed_acyclic_graph(self.graph)):
            self.graph.remove_edge(node1, node2)
            return False
        else:
            return True

    def set_ops(self, indexes):
        cursor = 0
        for b in self.blocks:
            num_nodes = b.num_nodes()
            b.set_ops(indexes[cursor:cursor+num_nodes])
            cursor += num_nodes

            self.graph.add_nodes_from(b.graph.nodes())
            self.graph.add_edges_from(b.graph.edges())

    def create_tensor(self, graph=None, train=True):
        if graph is None:
            graph = self.graph
        output_tensors = []
        for b in self.blocks:
            output_tensors.extend(b.create_tensor(graph, train=train))
        return self.output.create_tensor(output_tensors, train=train)

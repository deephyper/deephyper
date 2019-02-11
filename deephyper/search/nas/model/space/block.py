import networkx as nx

from deephyper.search.nas.model.space.node import Node


class Block:
    """This class represent a basic group of Nodes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.outputs = []
        self.inputs = []

    def num_nodes(self):
        return len(self.graph.nodes)

    @property
    def action_nodes(self):
        return self.graph.nodes

    def set_ops(self, indexes):
        middle_nodes = list(self.graph.nodes)
        for n in self.outputs:
            middle_nodes.remove(n)
        for n in self.inputs:
            middle_nodes.remove(n)
        assert len(indexes) == len(self.inputs) + len(self.outputs) + len(middle_nodes), (f'len(indexes): {len(indexes)} == len(self.inputs): {len(self.inputs)} + len(self.outputs): {len(self.outputs)} + len(middle_nodes): {len(middle_nodes)}')
        cursor = 0
        for n in self.inputs:
            n.set_op(indexes[cursor])
            cursor += 1
        for n in middle_nodes:
            n.set_op(indexes[cursor])
            cursor += 1
        for n in self.outputs:
            n.set_op(indexes[cursor])
            cursor += 1

    def add_node(self, node):
        assert isinstance(node, Node)
        assert not node in self.graph.nodes

        self.graph.add_node(node)

        self.outputs.append(node)
        self.inputs.append(node)

        return node

    def add_edge(self, node1, node2):
        '''Create a new edge in the block graph.

        The edge created corresponds to : node1 -> node2.

        Args:
            node1 (Node)
            node2 (Node)

        Return:
            (bool) True if the edge was successfully created, False if not.
        '''
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)

        self.graph.add_edge(node1, node2)

        if not(nx.is_directed_acyclic_graph(self.graph)):
            self.graph.remove_edge(node1, node2)
            return False
        else:
            if node1 in self.outputs:
                self.outputs.remove(node1)
            if node2 in self.inputs:
                self.inputs.remove(node2)
            return True

    def max_num_ops(self):
        mx = 0
        for n in list(self.graph.nodes):
            mx = max(mx, n.num_ops())
        return mx

    def create_tensor(self, graph=None, train=True):
        assert len(self.outputs) > 0
        if graph is None:
            graph = self.graph
        output_tensors = [create_tensor_aux(graph, n, train=train) for n in self.outputs]
        return output_tensors

def create_tensor_aux(g, n, train=None):
    """Recursive function to create the tensors from the graph.

    Args:
        g (nx.DiGraph): a graph
        n (nx.Node): a node
        train (bool): True if the network is built for training, False if the network is built for validation/testing (for example False will deactivate Dropout).

    Return:
        the tensor represented by n.
    """

    if n._tensor != None:
        output_tensor = n._tensor
    else:
        pred = list(g.predecessors(n))
        if len(pred) == 0:
            output_tensor = n.create_tensor(train=train)
        else:
            output_tensor = n.create_tensor([create_tensor_aux(g, s_i, train=train) for s_i in pred], train=train)
    return output_tensor

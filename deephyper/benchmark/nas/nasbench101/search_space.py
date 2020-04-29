import traceback

import networkx as nx
from collections.abc import Iterable

from deephyper.core.exceptions.nas.space import (
    NodeAlreadyAdded,
    StructureHasACycle,
    WrongSequenceToSetOperations,
)
from deephyper.benchmark.nas.nasbench101.nodes import ConstantNode, Node, VariableNode, OutputNode


class NxSearchSpace:
    """A NxSearchSpace is an search_space based on a networkx graph.
    """

    def __init__(self, seed=None, **kwargs):
        self.graph = nx.DiGraph()
        self.seed = seed

    def draw_graphviz(self, path):
        with open(path, "w") as f:
            try:
                nx.nx_agraph.write_dot(self.graph, f)
            except:
                print("Error: can't create graphviz file...")

    def __len__(self):
        """Number of VariableNodes in the current search_space.

        Returns:
            int: number of variable nodes in the current search_space.
        """

        return len(self.nodes)

    @property
    def nodes(self):
        """Nodes of the current KSearchSpace.

        Returns:
            iterator: nodes of the current KSearchSpace.
        """

        return list(self.graph.nodes)

    def add_node(self, node):
        """Add a new node to the search_space.

        Args:
            node (Node): node to add to the search_space.

        Raises:
            TypeError: if 'node' is not an instance of Node.
            NodeAlreadyAdded: if 'node' has already been added to the search_space.
        """

        if not isinstance(node, Node):
            raise TypeError(f"'node' argument should be an instance of Node!")

        if node in self.nodes:
            raise NodeAlreadyAdded(node)

        self.graph.add_node(node)

    def connect(self, node1, node2):
        """Create a new connection in the KSearchSpace graph.

        The edge created corresponds to : node1 -> node2.

        Args:
            node1 (Node)
            node2 (Node)

        Raise:
            StructureHasACycle: if the new edge is creating a cycle.
        """
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)

        self.graph.add_edge(node1, node2)

        if not (nx.is_directed_acyclic_graph(self.graph)):
            raise StructureHasACycle(
                f"the connection between {node1} -> {node2} is creating a cycle in the search_space's graph."
            )

    @property
    def size(self):
        """Size of the search space define by the search_space
        """
        s = 0
        for n in filter(lambda n: isinstance(n, VariableNode), self.nodes):
            if n.num_ops != 0:
                if s == 0:
                    s = n.num_ops
                else:
                    s *= n.num_ops
        return s

    @property
    def max_num_ops(self):
        """Returns the maximum number of operations accross all VariableNodes of the struct.

        Returns:
            int: maximum number of Operations for a VariableNode in the current Structure.
        """
        return max(map(lambda n: n.num_ops, self.variable_nodes))

    @property
    def num_nodes(self):
        """Returns the number of VariableNodes in the current Structure.

        Returns:
            int: number of VariableNodes in the current Structure.
        """
        return len(list(self.variable_nodes))

    @property
    def variable_nodes(self):
        """Iterator of VariableNodes of the search_space.

        Returns:
            (Iterator(VariableNode)): generator of VariablesNodes of the search_space.
        """
        return filter(lambda n: isinstance(n, VariableNode), self.nodes)

    def denormalize(self, indexes):
        """Denormalize a sequence of normalized indexes to get a sequence of absolute indexes. Useful when you want to compare the number of different search_spaces.

        Args:
            indexes (Iterable): a sequence of normalized indexes.

        Returns:
            list: A list of absolute indexes corresponding to operations choosen with relative indexes of `indexes`.
        """
        assert isinstance(
            indexes, Iterable
        ), 'Wrong argument, "indexes" should be of Iterable.'

        if len(indexes) != self.num_nodes:
            raise WrongSequenceToSetOperations(indexes, list(self.variable_nodes))

        return [
            vnode.denormalize(op_i) for op_i, vnode in zip(indexes, self.variable_nodes)
        ]

    def get_output_nodes(self):
        """Get nodes of 'graph' without successors.

        Return:
            list: the nodes without successors of a DiGraph.
        """
        nodes = list(self.graph.nodes())
        output_nodes = []
        for n in nodes:
            if len(list(self.graph.successors(n))) == 0:
                output_nodes.append(n)
        return output_nodes

    def set_ops(self, indexes):
        """Set the operations for each node of each cell of the search_space.

        Args:
            indexes (list):  element of list can be float in [0, 1] or int.

        Raises:
            WrongSequenceToSetOperations: raised when 'indexes' is of a wrong length.
        """
        if len(indexes) != len(list(self.variable_nodes)):
            raise WrongSequenceToSetOperations(indexes, list(self.variable_nodes))

        for op_i, node in zip(indexes, self.variable_nodes):
            node.set_op(op_i)

        output_nodes = self.get_output_nodes()

        self.output_node = self.set_output_node(self.graph, output_nodes)

    def set_output_node(self, graph, output_nodes):
        """Set the output node of the search_space.

        Args:
            graph (nx.DiGraph): graph of the search_space.
            output_nodes (Node): nodes of the current search_space without successors.

        Returns:
            Node: output node of the search_space.
        """
        out_node = OutputNode()
        for n in output_nodes:
            self.connect(n, out_node)
        return out_node

    def contract(self):
        to_contract = {"id", "connect"}
        graph = self.graph.copy()

#         none_nodes = [n for n in graph.nodes() if str(n) == "none"]
#         graph.remove_nodes_from(none_nodes)

        def left_to_contract(graph):
            node_names = set([str(n) for n in graph.nodes()])
            return len(node_names.intersection(to_contract)) > 0

        while left_to_contract(graph):
            for e_1, e_2 in graph.edges():
                if str(e_2) in to_contract:
                    graph = nx.contracted_edge(graph, (e_1, e_2), self_loops=False)
                    break
        return graph
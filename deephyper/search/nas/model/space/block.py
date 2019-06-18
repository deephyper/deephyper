import networkx as nx

from deephyper.search.nas.model.space.node import Node, VariableNode


class Block:
    """This class represent a basic group of Nodes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.outputs = list()
        self.inputs = list()

    def num_nodes(self):
        """Indicate the number of VariableNodes of the current Block.

        Returns:
            int: number of VariableNodes of the current Block.
        """

        return len(list(filter(lambda n: isinstance(n, VariableNode), self.nodes)))

    @property
    def action_nodes(self):
        """Return the list of VariableNodes in the current Block.

        Returns:
            list(VariableNode): list of VariableNodes of the current Block.
        """

        return filter(lambda n: isinstance(n, VariableNode), self.nodes)

    @property
    def size(self):
        s = 0
        for vn in self.action_nodes:
            vn_s = vn.num_ops
            if vn_s != 0:
                if s == 0:
                    s = vn_s
                else:
                    s *= vn_s
        return s

    @property
    def nodes(self):
        """Nodes of the current Block.

        Returns:
            iterator: nodes of the current Block.
        """

        return list(self.graph.nodes)

    def set_ops(self, indexes):
        """Set operations of VariableNodes of the current Block.

        Args:
            indexes (list): list of absolute (int) or relative (float) indexes to set operations of VariableNodes of the current Block.

        Raises:
            RuntimeError: ...
        """

        # nodes which are not inputs neither outputs
        mnodes = self.nodes
        for n in self.outputs:
            mnodes.remove(n)
        for n in self.inputs:
            if n in mnodes:
                mnodes.remove(n)

        variable_mnodes = list(
            filter(lambda n: isinstance(n, VariableNode), mnodes))
        variable_inputs = list(
            filter(lambda n: isinstance(n, VariableNode), self.inputs))
        variable_ouputs = list(
            filter(lambda n: isinstance(n, VariableNode), self.outputs))

        # number of VariableNodes in current Block
        nvariable_nodes = len(
            list(filter(lambda n: isinstance(n, VariableNode), self.nodes)))

        if len(indexes) != nvariable_nodes:
            raise RuntimeError(
                f'len(indexes) == {len(indexes)} when it should be {nvariable_nodes}')

        cursor = 0
        for n in variable_inputs:
            n.set_op(indexes[cursor])
            cursor += 1
        for n in variable_mnodes:
            n.set_op(indexes[cursor])
            cursor += 1
        for n in variable_ouputs:
            # Important if there is only one variable nodes
            if not n in variable_inputs:
                n.set_op(indexes[cursor])
                cursor += 1

    def denormalize(self, indexes):
        """Denormalize a sequence of normalized indexes to get a sequence of absolute indexes. Useful when you want to compare the number of different architectures.

        Args:
            indexes (Iterable): a sequence of normalized indexes.

        Returns:
            list: A list of absolute indexes corresponding to operations choosen with relative indexes of `indexes`.
        """
        den_list = []

        # nodes which are not inputs neither outputs
        mnodes = self.nodes
        for n in self.outputs:
            mnodes.remove(n)
        for n in self.inputs:
            if n in mnodes:
                mnodes.remove(n)

        variable_mnodes = list(
            filter(lambda n: isinstance(n, VariableNode), mnodes))
        variable_inputs = list(
            filter(lambda n: isinstance(n, VariableNode), self.inputs))
        variable_ouputs = list(
            filter(lambda n: isinstance(n, VariableNode), self.outputs))

        # number of VariableNodes in current Block
        nvariable_nodes = len(
            list(filter(lambda n: isinstance(n, VariableNode), self.nodes)))

        if len(indexes) != nvariable_nodes:
            raise RuntimeError(
                f'len(indexes) == {len(indexes)} when it should be {nvariable_nodes}')

        cursor = 0
        for n in variable_inputs:
            den_list.append(n.denormalize(indexes[cursor]))
            cursor += 1
        for n in variable_mnodes:
            den_list.append(n.denormalize(indexes[cursor]))
            cursor += 1
        for n in variable_ouputs:
            # Important if there is only one variable nodes
            if not n in variable_inputs:
                den_list.append(n.denormalize(indexes[cursor]))
                cursor += 1

        return den_list

    def add_node(self, node):
        """Add a new node to the current Block.

        Args:
            node (Node): the node to add to the current Block.

        Raises:
            RuntimeError: if the node to add is not an instance of Node .
            RuntimeError: if the node to add has already been added to the Block.
        """

        if not isinstance(node, Node):
            raise RuntimeError(f'node argument should be an instance of Node!')
        if node in self.nodes:
            raise RuntimeError(
                f'Node: {node} has already been added to the Block!')

        self.graph.add_node(node)

        self.outputs.append(node)
        self.inputs.append(node)

    def add_edge(self, node1, node2):
        """Create a new edge in the block graph.

        The edge created corresponds to : node1 -> node2.

        Args:
            node1 (Node)
            node2 (Node)

        Return:
            (bool) True if the edge was successfully created, False if not.
        """
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
        """Return the maximum number of operations for a VariableNode in current Block.

        Returns:
            int: maximum number of operations for a VariableNode in current Block.
        """

        mx = 0
        for n in filter(lambda n: isinstance(n, VariableNode), self.nodes):
            mx = max(mx, n.num_ops)
        return mx

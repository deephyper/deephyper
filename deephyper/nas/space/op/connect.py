from ..op import Operation


class Connect(Operation):
    """Connection node.

    Represents a possibility to create a connection between n1 -> n2.

    Args:
        graph (nx.DiGraph): a graph
        source_node (Node): source
    """

    def __init__(self, search_space, source_node, *args, **kwargs):
        self.search_space = search_space
        self.source_node = source_node
        self.destin_node = None

    def __str__(self):
        if type(self.source_node) is list:
            if len(self.source_node) > 0:
                ids = str(self.source_node[0].id)
                for n in self.source_node[1:]:
                    ids += ',' + str(n.id)
            else:
                ids = 'None'
        else:
            ids = self.source_node.id
        if self.destin_node is None:
            return f'{type(self).__name__}_{ids}->?'
        else:
            return f'{type(self).__name__}_{ids}->{self.destin_node.id}'

    def init(self, current_node):
        """Set the connection in the search_space graph from [n1] -> n2.
        """
        self.destin_node = current_node
        if type(self.source_node) is list:
            for n in self.source_node:
                self.search_space.connect(n, self.destin_node)
        else:
            self.search_space.connect(self.source_node, self.destin_node)

    def __call__(self, value, *args, **kwargs):
        return value

from deephyper.search.nas.model.space.op import Operation


class Connect(Operation):
    """Connection node.

    Represents a possibility to create a connection between n1 -> n2.

    Args:
        graph (nx.DiGraph): a graph
        n1 (Node): starting node
        n2 (Node): arrival node
    """

    def __init__(self, struct, n1, n2, *args, **kwargs):
        self.struct = struct
        self.n1 = n1
        self.n2 = n2

    def __str__(self):
        if type(self.n1) is list:
            if len(self.n1) > 0:
                ids = str(self.n1[0].id)
                for n in self.n1[1:]:
                    ids += ',' + str(n.id)
            else:
                ids = 'None'
        else:
            ids = self.n1.id
        return f'{type(self).__name__}_{ids}->{self.n2.id}'

    def init(self):
        """Set the connection in the structur graph from [n1] -> n2.
        """
        if type(self.n1) is list:
            for n in self.n1:
                self.struct.connect(n, self.n2)
        else:
            self.struct.connect(self.n1, self.n2)

    def __call__(self, value, *args, **kwargs):
        return value

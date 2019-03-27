import tensorflow as tf
from tensorflow import keras

import deephyper.search.nas.model.space.layers as deeplayers


class Operation:
    """Interface of an operation.
    """

    def __str__(self):
        return type(self).__name__

    def __call__(self, *args, **kwargs):
        """
        Returns:
            tensor: a tensor
        """
        raise NotImplementedError

    def is_set(self):
        """Preprocess the current operation.
        """


class Tensor(Operation):
    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor

    def __call__(self, *args, **kwargs):
        return self.tensor

class Connect(Operation):
    """Connection node.

    Represents a possibility to create a connection between n1 -> n2.

    Args:
        graph (nx.DiGraph): a graph
        n1 (Node): starting node
        n2 (Node): arrival node
    """
    def __init__(self, graph, n1, n2, *args, **kwargs):
        self.graph = graph
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

    def is_set(self):
        """Set the connection in the networkx graph.
        """
        if type(self.n1) is list:
            for n in self.n1:
                self.graph.add_edge(n, self.n2)
        else:
            self.graph.add_edge(self.n1, self.n2)

    def __call__(self, value, *args, **kwargs):
        return value


class AddByPadding(Operation):
    """Add operation.

    Args:
        graph:
        node (Node): current_node of the operation
        stacked_nodes (list(Node)): nodes to add
        axis (int): axis to concatenate
    """
    def __init__(self, graph=None, node=None, stacked_nodes=None, axis=-1):
        self.graph = graph
        self.node = node
        self.stacked_nodes = stacked_nodes
        self.axis = axis

    def is_set(self):
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.graph.add_edge(n, self.node)

    def __call__(self, values, **kwargs):
        values = values[:]
        max_len_shp = max([len(x.get_shape()) for x in values])

        # zeros padding
        if len(values) > 1:

            for i, v in enumerate(values):

                if len(v.get_shape()) < max_len_shp:
                    values[i] = keras.layers.Reshape(
                        (*tuple(v.get_shape()[1:]),
                        *tuple(1 for i in range(max_len_shp - len(v.get_shape())))
                        ))(v)

            max_dim_i = lambda i: max(map(lambda x: int(x.get_shape()[i]), values))
            max_dims = [None] + list(map(max_dim_i, range(1, max_len_shp)))

            paddings_dim_i = lambda i: list(map(lambda x: max_dims[i] - int(x.get_shape()[i]), values))
            paddings_dim = list(map(paddings_dim_i, range(1, max_len_shp)))

            for i in range(len(values)):
                paddings = list()
                for j in range(len(paddings_dim)):
                    p = paddings_dim[j][i]
                    lp = p // 2
                    rp = p - lp
                    paddings.append([lp, rp])
                if sum(map(sum,paddings)) != 0:
                    values[i] = deeplayers.Padding(paddings)(values[i])

        # concatenation
        if len(values) > 1:
            out = keras.layers.Add()(values)
        else:
            out = values[0]
        return out

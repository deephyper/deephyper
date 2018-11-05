import tensorflow as tf

class Operation:
    def __str__(self):
        return type(self).__name__

    def __call__(self, **kwargs):
        raise NotImplementedError

    def is_set(self):
        pass


class Tensor(Operation):
    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, **kwargs):
        return self.tensor


class Identity(Operation):
    def __call__(self, input, **kwargs):
        out = tf.identity(input)
        out = tf.reshape(out, tf.shape(out)[1:])
        return out


class Add(Operation):
    def __call__(self, inputs, **kwargs):
        return tf.add(inputs[0], inputs[1])


class Incr(Operation):
    def __call__(self, inputs, **kwargs):
        return tf.add(inputs[0], 1)


class Constant(Operation):
    def __init__(self, value):
        self.value = value

    def __call__(self, **kwargs):
        return tf.constant(self.value)


class Connect(Operation):
    """Connection node.

    Represents a possibility to create a connection between n1 -> n2.

    Args:
        graph (nx.DiGraph): a graph
        n1 (nx.Node): starting node
        n2 (nx.Node): arrival node
    """
    def __init__(self, graph, n1, n2):
        self.graph = graph
        self.n1 = n1
        self.n2 = n2

    def is_set(self):
        """Set the connection in the networkx graph.
        """
        self.graph.add_edge(self.n1, self.n2)

    def __call__(self, value, **kwargs):
        return value[0]


class Merge(Operation):
    """
        node = stack(stacked_nodes)
    """
    def __init__(self, graph, node, stacked_nodes, axis=0):

        self.graph = graph
        self.node = node
        self.stacked_nodes = stacked_nodes
        self.axis = axis

    def is_set(self):
        for n in self.stacked_nodes:
            self.graph.add_edge(n, self.node)

    def __call__(self, values, **kwargs):
        out = tf.concat(values, axis=self.axis)
        out = tf.layers.flatten(out)
        return out


def concat_last(inps):
        input_layer = inps[0]
        for i in range(1, len(inps)):
            curr_layer_shape = input_layer.get_shape().as_list()
            next_layer_shape = inps[i].get_shape().as_list()
            assert len(curr_layer_shape) == len(next_layer_shape), 'Concatenation of two tensors with different dimensions not supported.'
            max_shape = [ max(curr_layer_shape[x], next_layer_shape[x]) for x in range(len(curr_layer_shape))]
            curr_layer_padding_len = [[0,0]]
            next_layer_padding_len = [[0,0]]
            for d in range(1, len(max_shape[1:-1])+1):
                curr_layer_padding_len.append([
                    (max_shape[d] - curr_layer_shape[d]) // 2,
                    (max_shape[d] - curr_layer_shape[d]) -
                    ((max_shape[d] - curr_layer_shape[d]) // 2)])
                next_layer_padding_len.append(
                    [(max_shape[d] - next_layer_shape[d]) // 2,
                     (max_shape[d] - next_layer_shape[d]) -
                     ((max_shape[d] - next_layer_shape[d]) // 2)])
            curr_layer_padding_len.append([0,0])
            next_layer_padding_len.append([0,0])
            if sum([sum(x) for x in curr_layer_padding_len]) != 0:
                input_layer = tf.pad(input_layer, curr_layer_padding_len, 'CONSTANT')
            next_layer = inps[i]
            if sum([sum(x) for x in next_layer_padding_len]) != 0:
                next_layer = tf.pad(next_layer, next_layer_padding_len, 'CONSTANT')
            input_layer = tf.concat([input_layer, next_layer], len(max_shape)-1)
        return input_layer


class Concat(Operation):
    """Concat operation.
    """
    def __init__(self, graph=None, node=None, stacked_nodes=None, axis=0, last=False):
        self.graph = graph
        self.node = node
        self.stacked_nodes = stacked_nodes
        self.axis = axis
        self.last = last

    def is_set(self):
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.graph.add_edge(n, self.node)

    def __call__(self, values, **kwargs):
        if self.last:
            out = concat_last(values)
        else:
            out = tf.concat(values, axis=self.axis)
        return out

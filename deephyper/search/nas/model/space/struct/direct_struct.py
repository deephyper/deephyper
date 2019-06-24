from collections.abc import Iterable
from functools import reduce

import networkx as nx
from tensorflow import keras
from tensorflow.python.keras.utils.vis_utils import model_to_dot

from deephyper.core.exceptions.nas.struct import (InputShapeOfWrongType,
                                                  NodeAlreadyAdded,
                                                  StructureHasACycle,
                                                  WrongSequenceToSetOperations,
                                                  WrongOutputShape)
from deephyper.search.nas.model.space.node import (ConstantNode, Node,
                                                   VariableNode)
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.merge import Concatenate
from deephyper.search.nas.model.space.op.op1d import Identity
from deephyper.search.nas.model.space.struct import NxStructure


class DirectStructure(NxStructure):
    """A DirectStructure represents a search space of neural networks.

    >>> from tensorflow.keras.utils import plot_model
    >>> from deephyper.search.nas.model.space.struct import DirectStructure
    >>> from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
    >>> struct = DirectStructure((5, ), (1, ))
    >>> vnode = VariableNode()
    >>> struct.connect(struct.input_nodes[0], vnode)
    >>> vnode.add_op(Dense(10))
    >>> vnode.add_op(Dense(20))
    >>> output_node = ConstantNode(op=Dense(1))
    >>> struct.connect(vnode, output_node)
    >>> struct.set_ops([0])
    >>> model = struct.create_model()

    Args:
        input_shape (list(tuple(int))): list of shapes of all inputs.
        output_shape (tuple(int)): shape of output.

    Raises:
        InputShapeOfWrongType: [description]
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):

        super().__init__()

        if type(input_shape) is tuple:
            # we have only one input tensor here
            op = Tensor(keras.layers.Input(input_shape, name="input_0"))
            self.input_nodes = [ConstantNode(op=op, name='Input_0')]

        elif type(input_shape) is list and all(map(lambda x: type(x) is tuple, input_shape)):
            # we have a list of input tensors here
            self.input_nodes = list()
            for i in range(len(input_shape)):
                op = Tensor(keras.layers.Input(
                    input_shape[i], name=f"input_{i}"))
                inode = ConstantNode(op=op, name=f'Input_{i}')
                self.input_nodes.append(inode)
        else:
            raise InputShapeOfWrongType(input_shape)

        for node in self.input_nodes:
            self.graph.add_node(node)

        self.output_shape = output_shape
        self.output_node = None

        self._model = None

    def __len__(self):
        """Number of VariableNodes in the current structure.

        Returns:
            int: number of variable nodes in the current structure.
        """

        return len(self.nodes)

    @property
    def nodes(self):
        """Nodes of the current DirectStructure.

        Returns:
            iterator: nodes of the current DirectStructure.
        """

        return list(self.graph.nodes)

    def add_node(self, node):
        """Add a new node to the structure.

        Args:
            node (Node): node to add to the structure.

        Raises:
            TypeError: if 'node' is not an instance of Node.
            NodeAlreadyAdded: if 'node' has already been added to the structure.
        """

        if not isinstance(node, Node):
            raise TypeError(f"'node' argument should be an instance of Node!")

        if node in self.nodes:
            raise NodeAlreadyAdded(node)

        self.graph.add_node(node)

    def connect(self, node1, node2):
        """Create a new connection in the DirectStructure graph.

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

        if not(nx.is_directed_acyclic_graph(self.graph)):
            raise StructureHasACycle(
                f'the connection between {node1} -> {node2} is creating a cycle in the structure\'s graph.')

    @property
    def size(self):
        """Size of the search space define by the structure
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
    def depth(self):
        if self._model is None:
            raise RuntimeError(
                "Can't compute depth of model without creating a model.")
        return len(self.longest_path)

    @property
    def longest_path(self):
        if self._model is None:
            raise RuntimeError(
                "Can't compute longest path of model without creating a model.")
        nx_graph = nx.drawing.nx_pydot.from_pydot(model_to_dot(self._model))
        return nx.algorithms.dag.dag_longest_path(nx_graph)

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
        """Iterator of VariableNodes of the structure.

        Returns:
            (Iterator(VariableNode)): generator of VariablesNodes of the structure.
        """
        return filter(lambda n: isinstance(n, VariableNode), self.nodes)

    def set_ops(self, indexes):
        """Set the operations for each node of each cell of the structure.

        Args:
            indexes (list):  element of list can be float in [0, 1] or int.

        Raises:
            WrongSequenceToSetOperations: raised when 'indexes' is of a wrong length.
        """
        if len(indexes) != len(list(self.variable_nodes)):
            raise WrongSequenceToSetOperations(
                indexes, list(self.variable_nodes))

        for op_i, node in zip(indexes, self.variable_nodes):
            node.set_op(op_i)

        output_nodes = self.get_output_nodes()

        self.output_node = self.set_output_node(self.graph, output_nodes)

    def set_output_node(self, graph, output_nodes):
        """Set the output node of the structure.

        Args:
            graph (nx.DiGraph): graph of the structure.
            output_nodes (Node): nodes of the current structure without successors.

        Returns:
            Node: output node of the structure.
        """
        if len(output_nodes) == 1:
            node = ConstantNode(op=Identity(), name='Structure_Output')
            graph.add_node(node)
            graph.add_edge(output_nodes[0], node)
        else:
            node = ConstantNode(name='Structure_Output')
            node.set_op(Concatenate(self, node, output_nodes))
        return node

    def create_model(self):
        """Create the tensors corresponding to the structure.

        Returns:
            A keras.Model for the current structure with the corresponding set of operations.
        """

        output_tensor = self.create_tensor_aux(self.graph, self.output_node)
        if output_tensor.get_shape()[1:] != self.output_shape:
            raise WrongOutputShape(output_tensor, self.output_shape)

        input_tensors = [inode._tensor for inode in self.input_nodes]

        self._model = keras.Model(inputs=input_tensors, outputs=output_tensor)

        return keras.Model(inputs=input_tensors, outputs=output_tensor)

    def denormalize(self, indexes):
        """Denormalize a sequence of normalized indexes to get a sequence of absolute indexes. Useful when you want to compare the number of different architectures.

        Args:
            indexes (Iterable): a sequence of normalized indexes.

        Returns:
            list: A list of absolute indexes corresponding to operations choosen with relative indexes of `indexes`.
        """
        assert isinstance(
            indexes, Iterable), 'Wrong argument, "indexes" should be of Iterable.'

        if len(indexes) != self.num_nodes:
            raise WrongSequenceToSetOperations(
                indexes, list(self.variable_nodes))

        return [vnode.denormalize(op_i) for op_i, vnode in zip(indexes, self.variable_nodes)]

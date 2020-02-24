from collections.abc import Iterable
from functools import reduce

import networkx as nx
from tensorflow import keras
from tensorflow.python.keras.utils.vis_utils import model_to_dot

from deephyper.core.exceptions.nas.space import (
    InputShapeOfWrongType,
    NodeAlreadyAdded,
    StructureHasACycle,
    WrongOutputShape,
    WrongSequenceToSetOperations,
)
from deephyper.search.nas.model.space import NxSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, Node, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.merge import Concatenate
from deephyper.search.nas.model.space.op.op1d import Identity


class KSearchSpace(NxSearchSpace):
    """A KSearchSpace represents a search space of neural networks.

    >>> from tensorflow.keras.utils import plot_model
    >>> from deephyper.search.nas.model.space import KSearchSpace
    >>> from deephyper.search.nas.model.space.node import VariableNode, ConstantNode
    >>> from deephyper.search.nas.model.space.op.op1d import Dense
    >>> struct = KSearchSpace((5, ), (1, ))
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
            self.input_nodes = [ConstantNode(op=op, name="Input_0")]

        elif type(input_shape) is list and all(
            map(lambda x: type(x) is tuple, input_shape)
        ):
            # we have a list of input tensors here
            self.input_nodes = list()
            for i in range(len(input_shape)):
                op = Tensor(keras.layers.Input(input_shape[i], name=f"input_{i}"))
                inode = ConstantNode(op=op, name=f"Input_{i}")
                self.input_nodes.append(inode)
        else:
            raise InputShapeOfWrongType(input_shape)

        for node in self.input_nodes:
            self.graph.add_node(node)

        self.output_shape = output_shape
        self.output_node = None

        self._model = None

    @property
    def depth(self):
        if self._model is None:
            raise RuntimeError("Can't compute depth of model without creating a model.")
        return len(self.longest_path)

    @property
    def longest_path(self):
        if self._model is None:
            raise RuntimeError(
                "Can't compute longest path of model without creating a model."
            )
        nx_graph = nx.drawing.nx_pydot.from_pydot(model_to_dot(self._model))
        return nx.algorithms.dag.dag_longest_path(nx_graph)

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
        if len(output_nodes) == 1:
            # node = ConstantNode(op=Identity(), name='Structure_Output')
            # graph.add_node(node)
            # graph.add_edge(output_nodes[0], node)
            node = output_nodes[0]
        else:
            # node = ConstantNode(name='Structure_Output')
            # op = Concatenate(self, output_nodes)
            # node.set_op(op=op)
            node = output_nodes
        return node

    def create_model(self):
        """Create the tensors corresponding to the search_space.

        Returns:
            A keras.Model for the current search_space with the corresponding set of operations.
        """
        if type(self.output_node) is list:
            output_tensors = [
                self.create_tensor_aux(self.graph, out) for out in self.output_node
            ]

            for out_T in output_tensors:
                output_n = int(out_T.name.split("/")[0].split("_")[-1])
                out_S = self.output_shape[output_n]
                if out_T.get_shape()[1:] != out_S:
                    raise WrongOutputShape(out_T, out_S)

            input_tensors = [inode._tensor for inode in self.input_nodes]

            self._model = keras.Model(inputs=input_tensors, outputs=output_tensors)
        else:
            output_tensors = self.create_tensor_aux(self.graph, self.output_node)
            if output_tensors.get_shape()[1:] != self.output_shape:
                raise WrongOutputShape(output_tensors, self.output_shape)

            input_tensors = [inode._tensor for inode in self.input_nodes]

            self._model = keras.Model(inputs=input_tensors, outputs=[output_tensors])

        return self._model

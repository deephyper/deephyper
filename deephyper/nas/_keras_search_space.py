import copy
import logging
import warnings

import networkx as nx
import numpy as np
import tensorflow as tf
from deephyper.core.exceptions.nas.space import (
    InputShapeOfWrongType,
    WrongSequenceToSetOperations,
)
from deephyper.nas._nx_search_space import NxSearchSpace
from deephyper.nas.node import ConstantNode
from deephyper.nas.operation import Tensor
from tensorflow import keras
from tensorflow.python.keras.utils.vis_utils import model_to_dot

logger = logging.getLogger(__name__)


class KSearchSpace(NxSearchSpace):
    """A KSearchSpace represents a search space of neural networks.

    >>> import tensorflow as tf
    >>> from deephyper.nas import KSearchSpace
    >>> from deephyper.nas.node import ConstantNode, VariableNode
    >>> from deephyper.nas.operation import operation, Identity
    >>> Dense = operation(tf.keras.layers.Dense)
    >>> Dropout = operation(tf.keras.layers.Dropout)

    >>> class ExampleSpace(KSearchSpace):
    ...     def build(self):
    ...         # input nodes are automatically built based on `input_shape`
    ...         input_node = self.input_nodes[0]
    ...         # we want 4 layers maximum (Identity corresponds to not adding a layer)
    ...         for i in range(4):
    ...             node = VariableNode()
    ...             self.connect(input_node, node)
    ...             # we add 3 possible operations for each node
    ...             node.add_op(Identity())
    ...             node.add_op(Dense(100, "relu"))
    ...             node.add_op(Dropout(0.2))
    ...             input_node = node
    ...         output = ConstantNode(op=Dense(self.output_shape[0]))
    ...         self.connect(input_node, output)
    ...         return self
    ...
    >>>

    >>> space = ExampleSpace(input_shape=(1,), output_shape=(1,)).build()
    >>> space.sample().summary()

    Args:
        input_shape (list(tuple(int))): list of shapes of all inputs.
        output_shape (tuple(int)): shape of output.
        batch_size (list(tuple(int))): batch size of the input layer. If ``input_shape`` is defining a list of inputs, ``batch_size`` should also define a list of inputs.

    Raises:
        InputShapeOfWrongType: [description]
    """

    def __init__(
        self, input_shape, output_shape, batch_size=None, seed=None, *args, **kwargs
    ):

        super().__init__()

        self._random = np.random.RandomState(seed)

        self.input_shape = input_shape
        if type(input_shape) is tuple:
            # we have only one input tensor here
            op = Tensor(
                keras.layers.Input(input_shape, name="input_0", batch_size=batch_size)
            )
            self.input_nodes = [ConstantNode(op=op, name="Input_0")]

        elif type(input_shape) is list and all(
            map(lambda x: type(x) is tuple, input_shape)
        ):
            # we have a list of input tensors here
            self.input_nodes = list()
            for i in range(len(input_shape)):
                batch_size = batch_size[i] if type(batch_size) is list else None
                op = Tensor(
                    keras.layers.Input(
                        input_shape[i], name=f"input_{i}", batch_size=batch_size
                    )
                )
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
    def input(self):
        return self.input_nodes

    @property
    def output(self):
        return self.output_node

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

        :meta private:

        Args:
            indexes (list):  element of list can be float in [0, 1] or int.

        Raises:
            WrongSequenceToSetOperations: raised when 'indexes' is of a wrong length.
        """
        if len(indexes) != len(list(self.variable_nodes)):
            raise WrongSequenceToSetOperations(indexes, list(self.variable_nodes))

        for op_i, node in zip(indexes, self.variable_nodes):
            node.set_op(op_i)

        for node in self.mime_nodes:
            node.set_op()

        self.set_output_node()

    def create_model(self):
        """Create the tensors corresponding to the search_space.

        :meta private:

        Returns:
            A keras.Model for the current search_space with the corresponding set of operations.
        """
        # !the output layer does not have to be of the same shape as the data
        # !this depends on the loss
        if type(self.output_node) is list:
            output_tensors = [
                self.create_tensor_aux(self.graph, out) for out in self.output_node
            ]

            for out_T in output_tensors:
                output_n = int(out_T.name.split("/")[0].split("_")[-1])
                out_S = self.output_shape[output_n]
                if tf.keras.backend.is_keras_tensor(out_T):
                    out_T_shape = out_T.type_spec.shape
                    if out_T_shape[1:] != out_S:
                        warnings.warn(
                            f"The output tensor of shape {out_T_shape} doesn't match the expected shape {out_S}!",
                            RuntimeWarning,
                        )

            input_tensors = [inode._tensor for inode in self.input_nodes]

            self._model = keras.Model(inputs=input_tensors, outputs=output_tensors)
        else:
            output_tensors = self.create_tensor_aux(self.graph, self.output_node)
            if tf.keras.backend.is_keras_tensor(output_tensors):
                output_tensors_shape = output_tensors.type_spec.shape
                if output_tensors_shape[1:] != self.output_shape:
                    warnings.warn(
                        f"The output tensor of shape {output_tensors_shape} doesn't match the expected shape {self.output_shape}!",
                        RuntimeWarning,
                    )

            input_tensors = [inode._tensor for inode in self.input_nodes]

            self._model = keras.Model(inputs=input_tensors, outputs=[output_tensors])

        return self._model

    def choices(self):
        """Gives the possible choices for each decision variable of the search space.

        Returns:
            list: A list of tuple where each element corresponds to a discrete variable represented by ``(low, high)``.
        """
        return [(0, vnode.num_ops - 1) for vnode in self.variable_nodes]

    def sample(self, choice=None):
        """Sample a ``tf.keras.Model`` from the search space.

        Args:
            choice (list, optional): A list of decision for the operations of this search space. Defaults to None, will generate a random sample.

        Returns:
            tf.keras.Model: A Tensorflow Keras model.
        """

        if choice is None:
            choice = [self._random.randint(c[0], c[1] + 1) for c in self.choices()]

        self_copy = copy.deepcopy(self)
        self_copy.set_ops(choice)
        model = self_copy.create_model()

        return model

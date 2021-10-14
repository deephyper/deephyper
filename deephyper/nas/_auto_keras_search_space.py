import tensorflow as tf
from tensorflow import keras

from deephyper.nas._keras_search_space import KSearchSpace
from deephyper.nas.node import ConstantNode
from deephyper.nas.operation import Concatenate


class AutoKSearchSpace(KSearchSpace):
    """An AutoKSearchSpace represents a search space of neural networks.

    Args:
        input_shape (list(tuple(int))): list of shapes of all inputs.
        output_shape (tuple(int)): shape of output.
        regression (bool): if ``True`` the output will be a simple ``tf.keras.layers.Dense(output_shape[0])`` layer as the output layer. if ``False`` the output will be ``tf.keras.layers.Dense(output_shape[0], activation='softmax')``.

    Raises:
        InputShapeOfWrongType: [description]
    """

    def __init__(self, input_shape, output_shape, regression: bool, *args, **kwargs):
        super().__init__(input_shape, output_shape)
        self.regression = regression

    def set_output_node(self):
        """Set the output node of the search_space.

        Args:
            graph (nx.DiGraph): graph of the search_space.
            output_nodes (Node): nodes of the current search_space without successors.

        Returns:
            Node: output node of the search_space.
        """
        super().set_output_node()
        if type(self.output_node) is list:
            node = ConstantNode(name='OUTPUT_MERGE')
            op = Concatenate(self, self.output_node)
            node.set_op(op=op)
            self.output_node = node

    def create_model(self):
        """Create the tensors corresponding to the search_space.

        Returns:
            The output tensor.
        """
        if self.regression:
            activation = None
        else:
            activation = 'softmax'

        output_tensor = self.create_tensor_aux(self.graph, self.output_node)

        if len(output_tensor.get_shape()) > 2:
            output_tensor = keras.layers.Flatten()(output_tensor)

        output_tensor = keras.layers.Dense(
            self.output_shape[0], activation=activation,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))(output_tensor)

        input_tensors = [inode._tensor for inode in self.input_nodes]

        self._model = keras.Model(inputs=input_tensors, outputs=output_tensor)

        return keras.Model(inputs=input_tensors, outputs=output_tensor)

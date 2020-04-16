import tensorflow as tf

from ..space import AutoKSearchSpace
from ..space.node import VariableNode
from ..space.op.op1d import Dense, Identity


def create_search_space(
    input_shape=(2,),
    output_shape=(1,),
    num_layers=10,
    num_units=(1, 11),
    regression=True,
    **kwargs
):
    """Simple search space for a feed-forward neural network. No skip-connection. Looking over the number of units per layer and the number of layers.

    Args:
        input_shape (tuple, optional): True shape of inputs (no batch size dimension). Defaults to (2,).
        output_shape (tuple, optional): True shape of outputs (no batch size dimension).. Defaults to (1,).
        num_layers (int, optional): Maximum number of layers to have. Defaults to 10.
        num_units (tuple, optional): Range of number of units such as range(start, end, step_size). Defaults to (1, 11).
        regression (bool, optional): A boolean defining if the model is a regressor or a classifier. Defaults to True.

    Returns:
        AutoKSearchSpace: A search space object based on tf.keras implementations.
    """
    ss = AutoKSearchSpace(input_shape, output_shape, regression=True)

    prev_node = ss.input_nodes[0]

    for _ in range(num_layers):
        vnode = VariableNode()
        vnode.add_op(Identity())
        for i in range(*num_units):
            vnode.add_op(Dense(i, tf.nn.relu))

        ss.connect(prev_node, vnode)
        prev_node = vnode

    return ss

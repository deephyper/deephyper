import deephyper as dh
import tensorflow as tf

from ._base import Operation


class Concatenate(Operation):
    """Concatenate operation.

    Args:
        graph:
        node (Node):
        stacked_nodes (list(Node)): nodes to concatenate
        axis (int): axis to concatenate
    """

    def __init__(self, search_space, stacked_nodes=None, axis=-1):
        self.search_space = search_space
        self.node = None  # current_node of the operation
        self.stacked_nodes = stacked_nodes
        self.axis = axis

    def __str__(self):
        return "Concatenate"

    def init(self, current_node):
        self.node = current_node
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.search_space.connect(n, self.node)

    def __call__(self, values, **kwargs):
        # case where there is no inputs
        if len(values) == 0:
            return []

        len_shp = max([len(x.get_shape()) for x in values])

        if len_shp > 4:
            raise RuntimeError(
                f"This concatenation is for 2D or 3D tensors only but a {len_shp-1}D is passed!"
            )

        # zeros padding
        if len(values) > 1:

            if all(
                map(
                    lambda x: len(x.get_shape()) == len_shp
                    or len(x.get_shape()) == (len_shp - 1),
                    values,
                )
            ):  # all tensors should have same number of dimensions 2d or 3d, but we can also accept a mix of 2d en 3d tensors
                # we have a mix of 2d and 3d tensors so we are expanding 2d tensors to be 3d with last_dim==1
                for i, v in enumerate(values):
                    if len(v.get_shape()) < len_shp:
                        values[i] = tf.keras.layers.Reshape(
                            (*tuple(v.get_shape()[1:]), 1)
                        )(v)
                # for 3d tensors concatenation is applied along last dim (axis=-1), so we are applying a zero padding to make 2nd dimensions (ie. shape()[1]) equals
                if len_shp == 3:
                    max_len = max(map(lambda x: int(x.get_shape()[1]), values))
                    paddings = map(lambda x: max_len - int(x.get_shape()[1]), values)
                    for i, (p, v) in enumerate(zip(paddings, values)):
                        lp = p // 2
                        rp = p - lp
                        values[i] = tf.keras.layers.ZeroPadding1D(padding=(lp, rp))(v)
                # elif len_shp == 2 nothing to do
            else:
                raise RuntimeError(
                    f"All inputs of concatenation operation should have same shape length:\n"
                    f"number_of_inputs=={len(values)}\n"
                    f"shape_of_inputs=={[str(x.get_shape()) for x in values]}"
                )

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Concatenate(axis=-1)(values)
        else:
            out = values[0]
        return out


class AddByPadding(Operation):
    """Add operation. If tensor are of different shapes a padding will be applied before adding them.

    Args:
        search_space (KSearchSpace): [description]. Defaults to None.
        activation ([type], optional): Activation function to apply after adding ('relu', tanh', 'sigmoid'...). Defaults to None.
        stacked_nodes (list(Node)): nodes to add.
        axis (int): axis to concatenate.
    """

    def __init__(self, search_space, stacked_nodes=None, activation=None, axis=-1):
        self.search_space = search_space
        self.node = None  # current_node of the operation
        self.stacked_nodes = stacked_nodes
        self.activation = activation
        self.axis = axis

    def init(self, current_node):
        self.node = current_node
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.search_space.connect(n, self.node)

    def __call__(self, values, **kwargs):
        # case where there is no inputs
        if len(values) == 0:
            return []

        values = values[:]
        max_len_shp = max([len(x.get_shape()) for x in values])

        # zeros padding
        if len(values) > 1:

            for i, v in enumerate(values):

                if len(v.get_shape()) < max_len_shp:
                    values[i] = tf.keras.layers.Reshape(
                        (
                            *tuple(v.get_shape()[1:]),
                            *tuple(1 for i in range(max_len_shp - len(v.get_shape()))),
                        )
                    )(v)

            def max_dim_i(i):
                return max(map(lambda x: int(x.get_shape()[i]), values))

            max_dims = [None] + list(map(max_dim_i, range(1, max_len_shp)))

            def paddings_dim_i(i):
                return list(map(lambda x: max_dims[i] - int(x.get_shape()[i]), values))

            paddings_dim = list(map(paddings_dim_i, range(1, max_len_shp)))

            for i in range(len(values)):
                paddings = list()
                for j in range(len(paddings_dim)):
                    p = paddings_dim[j][i]
                    lp = p // 2
                    rp = p - lp
                    paddings.append([lp, rp])
                if sum(map(sum, paddings)) != 0:
                    values[i] = dh.layers.Padding(paddings)(values[i])

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Add()(values)
            if self.activation is not None:
                out = tf.keras.layers.Activation(self.activation)(out)
        else:
            out = values[0]
        return out


class AddByProjecting(Operation):
    """Add operation. If tensors are of different shapes a projection will be applied before adding them.

    Args:
        search_space (KSearchSpace): [description]. Defaults to None.
        activation ([type], optional): Activation function to apply after adding ('relu', tanh', 'sigmoid'...). Defaults to None.
        stacked_nodes (list(Node)): nodes to add.
        axis (int): axis to concatenate.
    """

    def __init__(self, search_space, stacked_nodes=None, activation=None, axis=-1):
        self.search_space = search_space
        self.node = None  # current_node of the operation
        self.stacked_nodes = stacked_nodes
        self.activation = activation
        self.axis = axis

    def init(self, current_node):
        self.node = current_node
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.search_space.connect(n, self.node)

    def __call__(self, values, seed=None, **kwargs):
        # case where there is no inputs
        if len(values) == 0:
            return []

        values = values[:]
        max_len_shp = max([len(x.get_shape()) for x in values])

        # projection
        if len(values) > 1:

            for i, v in enumerate(values):

                if len(v.get_shape()) < max_len_shp:
                    values[i] = tf.keras.layers.Reshape(
                        (
                            *tuple(v.get_shape()[1:]),
                            *tuple(1 for i in range(max_len_shp - len(v.get_shape()))),
                        )
                    )(v)

            proj_size = values[0].get_shape()[self.axis]

            for i in range(len(values)):
                if values[i].get_shape()[self.axis] != proj_size:
                    values[i] = tf.keras.layers.Dense(
                        units=proj_size,
                        kernel_initializer=tf.keras.initializers.glorot_uniform(
                            seed=seed
                        ),
                    )(values[i])

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Add()(values)
            if self.activation is not None:
                out = tf.keras.layers.Activation(self.activation)(out)
        else:
            out = values[0]
        return out

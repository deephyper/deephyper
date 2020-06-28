import spektral
import tensorflow as tf
from . import Operation
from numpy.random import seed
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.activations import tanh
from spektral.layers.ops import filter_dot
import tensorflow.keras.backend as K

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)


# TODO: add graph gathering, graph batch normalization

# SELF DEFINED
class Apply1DMask(tf.keras.layers.Layer):
    def __init__(self):
        super(Apply1DMask, self).__init__()
        pass

    def __str__(self):
        return f"Apply a 1D mask to the dense layers"

    def call(self, inputs, **kwargs):
        assert type(inputs) is list
        return tf.multiply(inputs[0], inputs[1])


class GraphIdentity(tf.keras.layers.Layer):
    def __init__(self):
        super(GraphIdentity, self).__init__()
        pass

    def __str__(self):
        return f"Identity of Node features"

    def call(self, inputs, **kwargs):
        return inputs[0]


class GraphMaxPool(tf.keras.layers.Layer):
    """
    A GraphPool gathers data from local neighborhoods of a graph.
    This layer does a max-pooling over the feature vectors of atoms in a
    neighborhood. You can think of this layer as analogous to a max-pooling layer
    for 2D convolutions but which operates on graphs instead.

    Adapted from: https://github.com/deepchem/deepchem/blob/master/deepchem/models/layers.py
    """

    def __init__(self, **kwargs):
        super(GraphMaxPool, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphMaxPool"

    def call(self, inputs, **kwargs):
        X = inputs[0]  # Node features (?, N, F)
        A = inputs[1]  # Adjacency matrix (?, N, N)

        # If tf 1.15, can use tf.repeat, <= 1.14 tf.keras.backend.repeat_elements
        X_shape = X.get_shape().as_list()  # [?, N, F]
        X_expand = tf.expand_dims(X, 1)  # [?, 1, N, F]
        X_expand = tf.keras.backend.repeat_elements(X_expand, X_shape[1],
                                                    axis=1)  # [?, N, N, F]

        A_expand = tf.expand_dims(A, 3)  # [?, N, N, 1]
        A_expand = tf.keras.backend.repeat_elements(A_expand, X_shape[2],
                                                    axis=3)  # [?, N, N, F]
        out_expand = tf.multiply(X_expand, A_expand)  # [?, N, N, F]

        return tf.reduce_max(out_expand, axis=1)  # [?, N, F]


class GraphSumPool(tf.keras.layers.Layer):
    """
    A GraphPool gathers data from local neighborhoods of a graph.
    This layer does a max-pooling over the feature vectors of atoms in a
    neighborhood. You can think of this layer as analogous to a max-pooling layer
    for 2D convolutions but which operates on graphs instead.

    Adapted from: https://github.com/deepchem/deepchem/blob/master/deepchem/models/layers.py
    """

    def __init__(self, **kwargs):
        super(GraphSumPool, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphSumPool"

    def call(self, inputs, **kwargs):
        X = inputs[0]  # Node features (?, N, F)
        A = inputs[1]  # Adjacency matrix (?, N, N)

        # If tf 1.15, can use tf.repeat, <= 1.14 tf.keras.backend.repeat_elements
        X_shape = X.get_shape().as_list()  # [?, N, F]
        X_expand = tf.expand_dims(X, 1)  # [?, 1, N, F]
        X_expand = tf.keras.backend.repeat_elements(X_expand, X_shape[1],
                                                    axis=1)  # [?, N, N, F]

        A_expand = tf.expand_dims(A, 3)  # [?, N, N, 1]
        A_expand = tf.keras.backend.repeat_elements(A_expand, X_shape[2],
                                                    axis=3)  # [?, N, N, F]
        out_expand = tf.multiply(X_expand, A_expand)  # [?, N, N, F]

        return tf.reduce_sum(out_expand, axis=1)  # [?, N, F]


class GraphMeanPool(tf.keras.layers.Layer):
    """
    A GraphPool gathers data from local neighborhoods of a graph.
    This layer does a max-pooling over the feature vectors of atoms in a
    neighborhood. You can think of this layer as analogous to a max-pooling layer
    for 2D convolutions but which operates on graphs instead.

    Adapted from: https://github.com/deepchem/deepchem/blob/master/deepchem/models/layers.py
    """

    def __init__(self, **kwargs):
        super(GraphMeanPool, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphMeanPool"

    def call(self, inputs, **kwargs):
        X = inputs[0]  # Node features (?, N, F)
        A = inputs[1]  # Adjacency matrix (?, N, N)

        # If tf 1.15, can use tf.repeat, <= 1.14 tf.keras.backend.repeat_elements
        X_shape = X.get_shape().as_list()  # [?, N, F]
        X_expand = tf.expand_dims(X, 1)  # [?, 1, N, F]
        X_expand = tf.keras.backend.repeat_elements(X_expand, X_shape[1],
                                                    axis=1)  # [?, N, N, F]

        A_expand = tf.expand_dims(A, 3)  # [?, N, N, 1]
        A_expand = tf.keras.backend.repeat_elements(A_expand, X_shape[2],
                                                    axis=3)  # [?, N, N, F]
        out_expand = tf.multiply(X_expand, A_expand)  # [?, N, N, F]

        return tf.reduce_mean(out_expand, axis=1)  # [?, N, F]


class GraphGather(tf.keras.layers.Layer):
    """
    A GraphGather layer pools node-level feature vectors to create a graph feature vector.
    Many graph convolutional networks manipulate feature vectors per
    graph-node. For a molecule for example, each node might represent an
    atom, and the network would manipulate atomic feature vectors that
    summarize the local chemistry of the atom. However, at the end of
    the application, we will likely want to work with a molecule level
    feature representation. The `GraphGather` layer creates a graph level
    feature vector by combining all the node-level feature vectors.

    Adapted from: https://github.com/deepchem/deepchem/blob/master/deepchem/models/layers.py
    """

    def __init__(self, **kwargs):
        super(GraphGather, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphGather"

    def call(self, inputs, **kwargs):
        inputs = inputs
        # Node features (?, N, F)
        return tf.nn.tanh(tf.reduce_sum(inputs, axis=1))  # [?, F]


class GraphBatchNorm(Operation):
    """
    Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean
    activation close to 0 and the activation standard deviation close to 1.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphBatchNorm"

    def __call__(self, inputs, **kwargs):
        inputs = inputs[0]
        # Node features (?, N, F)
        out = tf.keras.layers.BatchNormalization(axis=-1)(inputs)  # (?, N, F)
        return out


# FROM SPEKTRAL
class GraphConv2(Operation):
    def __init__(self, channels, activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, *args, **kwargs):
        self.channels = channels
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        out = spektral.layers.GraphConv(channels=self.channels,
                                        activation=self.activation,
                                        use_bias=self.use_bias,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer,
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        activity_regularizer=self.activity_regularizer,
                                        kernel_constraint=self.kernel_constraint,
                                        bias_constraint=self.bias_constraint,
                                        **self.kwargs
                                        )(inputs)
        return out


class ChebConv2(Operation):
    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"ChebConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.ChebConv(channels=self.channels,
                                       activation=self.activation,
                                       use_bias=self.use_bias,
                                       kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer,
                                       kernel_regularizer=self.kernel_regularizer,
                                       bias_regularizer=self.bias_regularizer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       bias_constraint=self.bias_constraint,
                                       **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GraphSageConv2(Operation):
    def __init__(self,
                 channels,
                 aggregate_method='mean',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.aggregate_method = aggregate_method
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphSageConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.GraphSageConv(channels=self.channels,
                                            aggregate_method=self.aggregate_method,
                                            activation=self.activation,
                                            use_bias=self.use_bias,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            bias_regularizer=self.bias_regularizer,
                                            activity_regularizer=self.activity_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_constraint=self.bias_constraint,
                                            **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class EdgeConditionedConv2(Operation):
    def __init__(self,
                 channels,
                 kernel_network=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.kernel_network = kernel_network
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphEdgedConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.EdgeConditionedConv(channels=self.channels,
                                                  kernel_network=self.kernel_network,
                                                  activation=self.activation,
                                                  use_bias=self.use_bias,
                                                  kernel_initializer=self.kernel_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  bias_regularizer=self.bias_regularizer,
                                                  activity_regularizer=self.activity_regularizer,
                                                  kernel_constraint=self.kernel_constraint,
                                                  bias_constraint=self.bias_constraint,
                                                  **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GraphAttention2(Operation):
    def __init__(self,
                 channels,
                 attn_heads=1,
                 concat_heads=True,
                 dropout_rate=0.5,
                 return_attn_coef=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.attn_kernel_initializer = attn_kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.attn_kernel_constraint = attn_kernel_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAttention {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list

        out = spektral.layers.GraphAttention(channels=self.channels,
                                             attn_heads=self.attn_heads,
                                             concat_heads=self.concat_heads,
                                             dropout_rate=self.dropout_rate,
                                             return_attn_coef=self.return_attn_coef,
                                             activation=self.activation,
                                             use_bias=self.use_bias,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer,
                                             attn_kernel_initializer=self.attn_kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer,
                                             attn_kernel_regularizer=self.attn_kernel_regularizer,
                                             activity_regularizer=self.activity_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             bias_constraint=self.bias_constraint,
                                             attn_kernel_constraint=self.attn_kernel_constraint,
                                             **self.kwargs)(inputs)
        return out


class GraphConvSkip2(Operation):
    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphConvSkip {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.GraphConvSkip(channels=self.channels,
                                            activation=self.activation,
                                            use_bias=self.use_bias,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            bias_regularizer=self.bias_regularizer,
                                            activity_regularizer=self.activity_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_constraint=self.bias_constraint,
                                            **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class ARMAConv2(Operation):
    def __init__(self,
                 channels,
                 order=1,
                 iterations=1,
                 share_weights=False,
                 gcn_activation='relu',
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.activation = activation
        self.gcn_activation = gcn_activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"ARMAConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.ARMAConv(channels=self.channels,
                                       iterations=self.iterations,
                                       order=self.order,
                                       share_weights=self.share_weights,
                                       activation=self.activation,
                                       gcn_activation=self.gcn_activation,
                                       dropout_rate=self.dropout_rate,
                                       use_bias=self.use_bias,
                                       kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer,
                                       kernel_regularizer=self.kernel_regularizer,
                                       bias_regularizer=self.bias_regularizer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       bias_constraint=self.bias_constraint,
                                       **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class APPNP2(Operation):
    def __init__(self,
                 channels,
                 alpha=0.2,
                 propagations=1,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.alpha = alpha
        self.propagations = propagations
        self.mlp_activation = mlp_activation
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"APPNP {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.APPNP(channels=self.channels,
                                    mlp_hidden=self.mlp_hidden,
                                    alpha=self.alpha,
                                    propagations=self.propagations,
                                    mlp_activation=self.mlp_activation,
                                    activation=self.activation,
                                    dropout_rate=self.dropout_rate,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    activity_regularizer=self.activity_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint,
                                    **self.kwargs)(inputs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GINConv2(Operation):
    def __init__(self,
                 channels,
                 mlp_channels=16,
                 n_hidden_layers=0,
                 epsilon=None,
                 mlp_activation='relu',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.mlp_channels = mlp_channels
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.mlp_activation = mlp_activation
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GINConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list

        out = spektral.layers.GINConv(channels=self.channels,
                                      mlp_channels=self.mlp_channels,
                                      n_hidden_layers=self.n_hidden_layers,
                                      epsilon=self.epsilon,
                                      mlp_activation=self.mlp_activation,
                                      activation=self.activation,
                                      use_bias=self.use_bias,
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      bias_regularizer=self.bias_regularizer,
                                      activity_regularizer=self.activity_regularizer,
                                      kernel_constraint=self.kernel_constraint,
                                      bias_constraint=self.bias_constraint,
                                      **self.kwargs)(inputs)

        return out


class GlobalAvgPool2(Operation):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAvgPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

        out = spektral.layers.GlobalAvgPool(**self.kwargs)(inputs)

        if out.get_shape()[-2].value is None and len(out.get_shape()) is 3:
            out = tf.transpose(out, perm=[1, 0, 2])

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GlobalMaxPool2(Operation):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphMaxPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.GlobalMaxPool(**self.kwargs)(inputs)

        if out.get_shape()[-2].value is None and len(out.get_shape()) is 3:
            out = tf.transpose(out, perm=[1, 0, 2])

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GlobalSumPool2(Operation):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphSumPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.GlobalSumPool(**self.kwargs)(inputs)
        if out.get_shape()[-2].value is None and len(out.get_shape()) is 3:
            out = tf.transpose(out, perm=[1, 0, 2])

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class GlobalAttentionPool2(Operation):
    def __init__(self, channels=32,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.channels = channels
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAttentionPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.GlobalSumPool(**self.kwargs)(inputs)
        if out.get_shape()[-2].value is None and len(out.get_shape()) is 3:
            out = tf.transpose(out, perm=[1, 0, 2])

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class TopKPool2(Operation):
    def __init__(self, ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"TopKPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.TopKPool(ratio=self.ratio,
                                       return_mask=self.return_mask,
                                       sigmoid_gating=self.sigmoid_gating,
                                       kernel_initializer=self.kernel_initializer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       kwargs=self.kwargs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class SAGPool2(Operation):
    def __init__(self, ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"SAGPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.SAGPool(ratio=self.ratio,
                                      return_mask=self.return_mask,
                                      sigmoid_gating=self.sigmoid_gating,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      activity_regularizer=self.activity_regularizer,
                                      kernel_constraint=self.kernel_constraint,
                                      **self.kwargs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class MinCutPool2(Operation):
    def __init__(self,
                 k,
                 h=None,
                 return_mask=True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.k = k
        self.h = h
        self.return_mask = return_mask
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"MinCutPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.MinCutPool(k=self.k,
                                         h=self.h,
                                         return_mask=self.return_mask,
                                         activation=self.activation,
                                         use_bias=self.use_bias,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer,
                                         kernel_regularizer=self.kernel_regularizer,
                                         bias_regularizer=self.bias_regularizer,
                                         activity_regularizer=self.activity_regularizer,
                                         kernel_constraint=self.kernel_constraint,
                                         bias_constraint=self.bias_constraint,
                                         **self.kwargs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


class DiffPool2(Operation):
    def __init__(self,
                 k,
                 channels=None,
                 return_mask=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.k = k
        self.channels = channels
        self.return_mask = return_mask
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.kwargs = kwargs

    def __str__(self):
        return f"DiffPool"

    def __call__(self, inputs, **kwargs):
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.DiffPool(k=self.k,
                                       channels=self.channels,
                                       activation=self.activation,
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self.kernel_regularizer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       **self.kwargs)

        print(f"Output Tensor shape: {out.get_shape()} \n")
        return out


def aggregate(name, X, A):
    X_shape = X.get_shape().as_list()  # [?, N, C]
    X_expand = tf.expand_dims(X, 1)  # [?, 1, N, C]
    X_expand = tf.keras.backend.repeat_elements(X_expand, X_shape[1],
                                                axis=1)  # [?, N, N, C]

    A_expand = tf.expand_dims(A, 3)  # [?, N, N, 1]
    A_expand = tf.keras.backend.repeat_elements(A_expand, X_shape[2],
                                                axis=3)  # [?, N, N, C]
    out_expand = tf.multiply(X_expand, A_expand)  # [?, N, N, C]

    if name == 'maxpooling':
        return tf.reduce_max(out_expand, axis=1)  # [?, N, C]
    elif name == 'summation':
        return tf.reduce_sum(out_expand, axis=1)  # [?, N, C]
    elif name == 'mean':
        return tf.reduce_mean(out_expand, axis=1)  # [?, N, C]


class GraphAttentionCell(tf.keras.layers.Layer):
    # Dimension, Attention, Head, Aggregate, Combine, Activation
    def __init__(self,
                 channels,
                 attn_method='gat',
                 aggr_method='maxpooling',
                 attn_heads=1,
                 combine='identity',
                 concat_heads=True,
                 dropout_rate=0.5,
                 return_attn_coef=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super(GraphAttentionCell, self).__init__()
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.channels = channels
        self.attn_method = attn_method
        self.aggr_method = aggr_method
        self.combine = combine
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if concat_heads:
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(input_dim, self.channels),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))  # shape: (F, C)
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.channels,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))  # shape: (C, )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.channels, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head))  # shape: (C, 1)
            attn_kernel_neighs = self.add_weight(shape=(self.channels, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))  # shape: (C, 1)
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        if self.combine == 'mlp':
            if self.concat_heads:
                self.combine_kernel_1 = self.add_weight(shape=(2 * self.channels * self.attn_heads, 128),
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint,
                                                        name='combine_kernel_1')  # shape: (C*Head, 128)
                self.combine_kernel_2 = self.add_weight(shape=(128, 2 * self.channels * self.attn_heads),
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint,
                                                        name='combine_kernel_1')  # shape: (128, 2*C)
            else:
                self.combine_kernel_1 = self.add_weight(shape=(2 * self.channels, 128),
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint,
                                                        name='combine_kernel_1')  # shape: (C, 128)
                self.combine_kernel_2 = self.add_weight(shape=(128, 2 * self.channels),
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint,
                                                        name='combine_kernel_1')  # shape: (128, 2*C)
            if self.use_bias:
                self.combine_bias_1 = self.add_weight(shape=(128,),
                                                      initializer=self.bias_initializer,
                                                      regularizer=self.bias_regularizer,
                                                      constraint=self.bias_constraint,
                                                      name='combine_bias_1')  # shape: (128, )
                if self.concat_heads:
                    self.combine_bias_2 = self.add_weight(shape=(2 * self.channels * self.attn_heads),
                                                          initializer=self.bias_initializer,
                                                          regularizer=self.bias_regularizer,
                                                          constraint=self.bias_constraint,
                                                          name='combine_bias_2')  # shape: (128, )
                else:
                    self.combine_bias_2 = self.add_weight(shape=(2 * self.channels),
                                                          initializer=self.bias_initializer,
                                                          regularizer=self.bias_regularizer,
                                                          constraint=self.bias_constraint,
                                                          name='combine_bias_2')  # shape: (128, )
        self.built = True

    def call(self, inputs, **kwargs):
        assert self.attn_method in ['gat', 'sym-gat', 'gcn', 'const']
        X = inputs[0]  # shape: (?, N, F)
        A = inputs[1]  # shape: (?, N, N)

        outputs = []
        output_attn = []
        embeddings = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # shape: (F, C)
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
            # shape: [(C, 1), (C, 1)]

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # shape: (?, N, C)
            embeddings.append(features)  # shape: [(?, N, C), (?, N, C), ...]

            if self.attn_method == 'gat' or self.attn_method == 'sym-gat':
                # Compute attention coefficients
                # [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
                attn_for_self = K.dot(features, attention_kernel[0])  # [a_1]^T [Wh_i] # shape: (?, N, 1)
                attn_for_neighs = K.dot(features, attention_kernel[1])  # [a_2]^T [Wh_j] # shape: (?, N, 1)
                attn_for_neighs_T = K.permute_dimensions(attn_for_neighs, (0, 2, 1))  # shape: (?, 1, N)
                attn_coef = attn_for_self + attn_for_neighs_T  # shape: (?, N, N)
                attn_coef = LeakyReLU(alpha=0.2)(attn_coef)  # shape: (?, N, N)

                # Mask values before activation (Vaswani et al., 2017)
                mask = -10e9 * (1.0 - A)  # shape: (?, N, N)
                attn_coef += mask  # shape: (?, N, N)

                # Apply softmax to get attention coefficients
                attn_coef = K.softmax(attn_coef)  # shape: (?, N, N)
                output_attn.append(attn_coef)  # shape: [(?, N, N), (?, N, N), ...]

                # Apply dropout to attention coefficients
                attn_coef = Dropout(self.dropout_rate)(attn_coef)  # shape: (?, N, N)
                if self.attn_method == 'sym-gat':
                    attn_coef = attn_coef + K.permute_dimensions(attn_coef, (0, 2, 1))  # shape: (?, N, N)
            elif self.attn_method == 'const':
                attn_coef = A  # shape: (?, N, N)
            elif self.attn_method == 'gcn':
                D = tf.reduce_sum(A, axis=-1)  # node degrees shape: (?, N)
                D = tf.expand_dims(D, axis=-1)  # shape: (?, N, 1)
                Dt = K.permute_dimensions(D, (0, 2, 1))  # shape: (?, 1, N)
                D = tf.sqrt(tf.multiply(D, Dt))  # shape: (?, N, N)
                attn_coef = tf.math.divide_no_nan(A, D)  # shape: (?, N, N)

            # Convolution
            features = filter_dot(attn_coef, features)  # shape: (?, N, C)
            if self.use_bias:
                features = K.bias_add(features, self.biases[head])  # shape: (?, N, C)

            # Add output of attention head to final output
            outputs.append(features)  # shape: [(?, N, C), (?, N, C), ...]

        # Aggregate the heads' output according to the reduction method
        if self.concat_heads:
            output = K.concatenate(outputs)  # shape: (?, N, Head*C)
            embeddings = K.concatenate(embeddings)  # shape: (?, N, Head*C)
        else:
            output = K.mean(K.stack(outputs), axis=0)  # shape: (?, N, C)
            embeddings = K.mean(K.stack(embeddings), axis=0)  # shape: (?, N, C)

        output = aggregate(self.aggr_method, output, A)  # shape: (?, N, C) or (?, N, Head*C)

        # Combine local embedding with attention results
        output = K.concatenate((embeddings, output))  # shape: (?, N, 2*C) or (?, N, 2*Head*C)
        if self.combine == 'mlp':
            output = K.dot(output, self.combine_kernel_1)  # shape: (?, N, 128)
            if self.use_bias:
                output = K.bias_add(output, self.combine_bias_1)
            output = K.dot(output, self.combine_kernel_2)  # shape: (?, N, 2*C) or (?, N, 2*Head*C)
            if self.use_bias:
                output = K.bias_add(output, self.combine_bias_2)

        output = self.activation(output)  # shape: (?, N, 2*C) or (?, N, 2*Head*C)

        if self.return_attn_coef:
            return output, output_attn
        else:
            return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.output_dim,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'attn_heads': self.attn_heads,
            'concat_heads': self.concat_heads,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

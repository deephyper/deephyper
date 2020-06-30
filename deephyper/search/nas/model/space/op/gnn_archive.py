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

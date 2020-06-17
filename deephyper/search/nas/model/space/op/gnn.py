import spektral
import tensorflow
from . import Operation


class Apply1DMask(Operation):
    def __init__(self):
        pass

    def __str__(self):
        return f"Apply a 1D mask to the dense layers"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        return tensorflow.multiply(inputs[0], inputs[1])


class GraphIdentity(Operation):
    def __init__(self):
        pass

    def __str__(self):
        return f"Identity of Node features"

    def __call__(self, inputs, **kwargs):
        return inputs[0]


class GraphConv2(Operation):
    def __init__(self, channels, activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
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
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")
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
        print(f"Output Tensor shape: {out.get_shape()} \n")
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
        return f"GraphEdgedConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
        assert type(inputs) is list
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

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
        print(f"Output Tensor shape: {out.get_shape()} \n")
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
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]} \n")

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

        print(f"Output Tensor shape: {out.get_shape()} \n")
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
            out = tensorflow.transpose(out, perm=[1, 0, 2])

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
            out = tensorflow.transpose(out, perm=[1, 0, 2])

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
            out = tensorflow.transpose(out, perm=[1, 0, 2])

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
            out = tensorflow.transpose(out, perm=[1, 0, 2])

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


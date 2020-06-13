import spektral
import tensorflow
from . import Operation


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
        self.supports_masking = False
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphConv {self.channels} channels"

    def __call__(self, inputs, **kwargs):
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
        # print(f"Output shape: {[out[i].shape for i in range(len(out))]}")
        return out


class EdgeConditionedConv2(Operation):
    def __init__(self, channels, kernel_network=None, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
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
        print(f"Input shape: {[inputs[i].shape for i in range(len(inputs))]}")

        out = spektral.layers.EdgeConditionedConv(channels=self.channels,
                                                  kernel_network=self.kernel_network,
                                                  activation=self.activation,
                                                  use_bias=self.use_bias,
                                                  kernel_initializer=self.kernel_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  bias_regularizer=self.bias_regularizer,
                                                  activity_regularizer=self.activity_regularizer,
                                                  bias_constraint=self.bias_constraint,
                                                  **self.kwargs)(inputs)
        return out

import tensorflow as tf
from numpy.random import seed
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import LeakyReLU, Dropout, Dense
import tensorflow.keras.backend as K

seed(2020)
from tensorflow import set_random_seed

set_random_seed(2020)


def tensor_dot(X):
    assert type(X) is list
    out = K.dot(X[0], X[1])
    if len(X) > 2:
        for i in range(2, len(X)):
            out = K.dot(out, X[i])
    return out


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


class GlobalAvgPool(tf.keras.layers.Layer):
    def __init__(self, axis=-2, **kwargs):
        super(GlobalAvgPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAvgPool"

    def call(self, inputs, **kwargs):
        return tf.reduce_mean(inputs, axis=self.axis)


class GlobalMaxPool(tf.keras.layers.Layer):
    def __init__(self, axis=-2, **kwargs):
        super(GlobalMaxPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphMaxPool"

    def call(self, inputs, **kwargs):
        return tf.reduce_max(inputs, axis=self.axis)


class GlobalSumPool(tf.keras.layers.Layer):
    def __init__(self, axis=-2, **kwargs):
        super(GlobalSumPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphSumPool"

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(inputs, axis=self.axis)


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
    def __init__(self,
                 channels,
                 attn_method='gat',
                 aggr_method='maxpooling',
                 attn_heads=1,
                 combine='identity',
                 combine_dim=32,
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
        self.channels = channels
        self.attn_method = attn_method
        self.aggr_method = aggr_method
        self.combine = combine
        self.combine_dim = combine_dim
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

        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        self.combine_kernel = []  # Combine MLP kernels
        self.combine_bias = []  # Combine MLP bias

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads * 2
        else:
            self.output_dim = self.channels * 2

        assert self.attn_method in ['gat', 'sym-gat', 'gcn', 'const']
        assert self.aggr_method in ['maxpooling', 'sum', 'mean']
        assert self.combine in ['identity', 'mlp']

    def _build_kernel(self, input_dim, output_dim, name):
        kernel = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name=name)
        return kernel

    def _build_bias(self, output_dim, name):
        bias = self.add_weight(shape=(output_dim,),
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               name=name)
        return bias

    def _build_attn_kernel(self, input_dim, output_dim, name):
        kernel = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=self.attn_kernel_initializer,
                                 regularizer=self.attn_kernel_regularizer,
                                 constraint=self.attn_kernel_constraint,
                                 name=name)
        return kernel

    def build(self, input_shape):
        assert len(input_shape) >= 2 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            # shape: (F, C)
            kernel = self._build_kernel(input_dim, self.channels, f'kernel_{head}')
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                # shape: (C, )
                bias = self._build_bias(self.channels, f'bias_{head}')
                self.biases.append(bias)

            # Attention kernels
            if self.attn_method == 'gat' or self.attn_method == 'sym-gat':
                # shape: (C, 1)
                attn_kernel_self = self._build_attn_kernel(self.channels, 1, f'attn_kernel_self_{head}')
                attn_kernel_neighs = self._build_attn_kernel(self.channels, 1, f'attn_kernel_neigh_{head}')
                self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        if self.combine == 'mlp':
            # shape: (C*Head, 32)
            combine_kernel_1 = self._build_kernel(self.output_dim, self.combine_dim, 'combine_kernel_1')
            # shape: (32, 2*C)
            combine_kernel_2 = self._build_kernel(self.combine_dim, self.output_dim, 'combine_kernel_2')

            self.combine_kernel.append([combine_kernel_1, combine_kernel_2])
            if self.use_bias:
                # shape: (32, )
                combine_bias_1 = self._build_bias(self.combine_dim, 'combine_bias_1')
                combine_bias_2 = self._build_bias(self.output_dim, 'combine_bias_2')
                self.combine_bias.append([combine_bias_1, combine_bias_2])

        self.built = True

    def call(self, inputs, **kwargs):
        X = inputs[0]  # shape: (?, N, F)
        A = inputs[1]  # shape: (?, N, N)

        outputs = []
        output_attn = []
        embeddings = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # shape: (F, C)
            # shape: [(C, 1), (C, 1)]

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # shape: (?, N, C)
            embeddings.append(features)  # shape: [(?, N, C), (?, N, C), ...]

            if self.attn_method == 'gat' or self.attn_method == 'sym-gat':
                attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
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
            features = K.batch_dot(attn_coef, features)  # shape: (?, N, C)
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
            output = K.dot(output, self.combine_kernel[0][0])  # shape: (?, N, 32)
            if self.use_bias:
                output = K.bias_add(output, self.combine_bias[0][0])
            output = K.dot(output, self.combine_kernel[0][1])  # shape: (?, N, 2*C) or (?, N, 2*Head*C)
            if self.use_bias:
                output = K.bias_add(output, self.combine_bias[0][1])

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
            'attn_method': self.attn_method,
            'aggr_method': self.aggr_method,
            'combine': self.combine,
            'combine_dim': self.combine_dim,
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


# noinspection PyAttributeOutsideInit
class GraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(GraphConv, self).__init__()
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def _build_kernel(self, input_dim, output_dim, name):
        kernel = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name=name)
        return kernel

    def _build_bias(self, output_dim, name):
        bias = self.add_weight(shape=(output_dim,),
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               name=name)
        return bias

    def build(self, input_shape):
        assert len(input_shape) >= 2 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])
        self.kernel = self._build_kernel(input_dim, self.channels, 'GraphConv_kernel')
        if self.use_bias:
            self.bias = self._build_bias(self.channels, 'GraphConv_bias')
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        X = inputs[0]
        A = inputs[1]
        B, N, F = K.int_shape(X)
        Ah = A + tf.eye(N, batch_shape=[B])
        D = tf.reduce_sum(Ah, axis=-1)
        Dh = tf.linalg.diag(tf.pow(D, -0.5))

        # Graph Convolution
        # Z = D^{-1/2}(A+I)D^{-1/2}XW+b
        output = tensor_dot([Dh, Ah, Dh, X, self.kernel])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# noinspection PyAttributeOutsideInit
class ChebConv(GraphConv):
    def __init__(self,
                 channels,
                 K,
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
        super(ChebConv, self).__init__(**kwargs)
        self.channels = channels
        self.K = K
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])
        self.kernel = self._build_kernel(input_dim, self.channels, 'CheBConv_kernel')
        if self.use_bias:
            self.bias = self._build_bias(self.channels, 'ChebCONV_bias')
        else:
            self.bias = None
        self.built = True


class GATMPNN(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 message_fn='ecgcn',
                 hidden=64,
                 kernel_network=None,
                 update_fn='MLP',
                 attn_method='sym-gat',
                 attn_heads=2,
                 dropout_rate=0.5,
                 aggr_method='max',
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        GATMPNN (graph attention message passing neural networks)
        Args:
            channels: int
                number of output channels
            message_fn: str, optional
                message function in the model
            update_fn: str, optional
                update function in the model
        """
        super(GATMPNN, self).__init__()
        self.channels = channels
        self.message_fn = message_fn
        self.hidden = hidden
        self.kernel_network = kernel_network
        self.update_fn = update_fn
        self.attn_method = attn_method
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.aggr_method = aggr_method
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if self.message_fn == 'ecgcn':
            self.message_function = EdgeConv(self.channels)
        self._trainable_weights += self.message_function.trainable_weights
        if self.update_fn == 'MLP':
            self.update_function = MLP(self.channels)
        self._trainable_weights += self.update_function.trainable_weights
        super(GATMPNN, self).build(input_shape)
        return

    def call(self, inputs, **kwargs):
        X, A, E = inputs
        out = self.message_function([X, A, E])
        out = self.update_function([X, out])
        return out

    def _build_kernel(self, input_dim, output_dim, name):
        kernel = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name=name)
        return kernel

    def _build_bias(self, output_dim, name):
        bias = self.add_weight(shape=(output_dim,),
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               name=name)
        return bias


class EdgeConv(GATMPNN):
    def __init__(self, channels, **kwargs):
        super(EdgeConv, self).__init__(channels, **kwargs)
        self.channels = channels
        self.kernel_network_layers = []

    def build(self, input_shape):
        assert len(input_shape) >= 3 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])
        # BUILD KERNEL NETWORK
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                self.kernel_network_layers.append(Dense(l,
                                                        name=f'Edge_MLP_{i}',
                                                        activation=self.activation,
                                                        use_bias=self.use_bias,
                                                        kernel_initializer=self.kernel_initializer,
                                                        bias_initializer=self.bias_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        bias_constraint=self.bias_constraint
                                                        ))
        self.kernel_network_layers.append(Dense(self.channels * input_dim, name='Edge_MLP_out'))
        self.edge_bias = self._build_bias(self.channels, name='Edge_bias')
        self.gat_network = GAT(self.channels)
        self.build = True

    def call(self, inputs, **kwargs):
        X, A, E = inputs
        B, N, F = K.int_shape(X)

        # FILTER NETWORK
        kernel_network = E
        for l in self.kernel_network_layers:
            kernel_network = l(kernel_network)

        # CONVOLUTION
        target_shape = (-1, N, N, self.channels, F)
        kernel = K.reshape(kernel_network, target_shape)
        conv_out = kernel * A[..., None, None]

        # AGGREGATE METHOD
        output = tf.einsum('abicf,aif->abic', conv_out, X)
        if self.aggr_method == 'sum':
            output = tf.reduce_sum(output, axis=-2)
        elif self.aggr_method == 'mean':
            output = tf.reduce_mean(output, axis=-2)
        elif self.aggr_method == 'max':
            output = tf.reduce_max(output, axis=-2)

        output = self.gat_network([X, A, output])

        if self.use_bias:
            output = K.bias_add(output, self.edge_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class GAT(GATMPNN):
    def __init__(self, channels, **kwargs):
        super(GAT, self).__init__(channels, **kwargs)
        self.channels = channels
        self.root_kernels = []
        self.root_biases = []
        self.attn_kernels = []

    def build(self, input_shape):
        assert len(input_shape) >= 3 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])

        # BUILD ATTN ROOT KERNEL
        for head in range(self.attn_heads):
            root_kernel = self._build_kernel(input_dim, self.channels, name=f'Root_kernel_{head}')
            self.root_kernels.append(root_kernel)
            if self.use_bias:
                root_bias = self._build_bias(self.channels, name=f'Root_bias_{head}')
                self.root_biases.append(root_bias)
            if self.attn_method in ['gat', 'sym-gat']:
                attn_kernel_self = self._build_kernel(self.channels, 1, f'Attn_kernel_self_{head}')
                attn_kernel_nbrs = self._build_kernel(self.channels, 1, f'Attn_kernel_nbrs_{head}')
                self.attn_kernels.append([attn_kernel_self, attn_kernel_nbrs])
        self.build = True

    def call(self, inputs, **kwargs):
        X, A, M = inputs
        # ROOT MULTIPLICATION
        root_outputs = []
        for head in range(self.attn_heads):
            root_kernel = self.root_kernels[head]
            root_output = K.dot(X, root_kernel)
            if self.attn_method in ['gat', 'sym-gat']:
                attn_kernel = self.attn_kernels[head]
                attn_for_self = K.dot(root_output, attn_kernel[0])
                attn_for_nbrs = K.dot(root_output, attn_kernel[1])
                attn_for_nbrs_T = K.permute_dimensions(attn_for_nbrs, (0, 2, 1))
                attn_coef = attn_for_self + attn_for_nbrs_T
                attn_coef = LeakyReLU(alpha=0.2)(attn_coef)
                mask = -10e9 * (1.0 - A)
                attn_coef += mask
                attn_coef = K.softmax(attn_coef)
                attn_coef = Dropout(self.dropout_rate)(attn_coef)
                if self.attn_method == 'sym-gat':
                    attn_coef = attn_coef + K.permute_dimensions(attn_coef, (0, 2, 1))
            elif self.attn_method == 'const':
                attn_coef = A
            elif self.attn_method == 'gcn':
                D = tf.reduce_sum(A, axis=-1)
                D = tf.expand_dims(D, axis=-1)
                Dt = K.permute_dimensions(D, (0, 2, 1))
                D = tf.sqrt(tf.multiply(D, Dt))
                attn_coef = tf.math.divide_no_nan(A, D)
            root_output = K.batch_dot(attn_coef, root_output)
            if self.use_bias:
                root_output = K.bias_add(root_output, self.root_biases[head])
            root_outputs.append(root_output)
        root_outputs = K.mean(K.stack(root_outputs), axis=0)
        output = root_outputs + M

        return output


class MLP(GATMPNN):
    def __init__(self, channels, **kwargs):
        super(MLP, self).__init__(channels, **kwargs)
        self.channels = channels

    def build(self, input_shape):
        assert len(input_shape) >= 2 and type(input_shape) is list
        input_dim = int(input_shape[0][-1])
        self.MLP_W = self._build_kernel(input_dim, self.channels, 'MLP_W')
        self.MLP_b = self._build_bias(self.channels, 'MLP_bias')
        self.build = True

    def call(self, inputs, **kwargs):
        X, M = inputs
        output = K.dot(X, self.MLP_W) + M
        if self.use_bias:
            output = K.bias_add(output, self.MLP_b)
        if self.activation is not None:
            output = self.activation(output)
        return output





#### SPARSE MPNN
class SPARSE_MPNN(tf.keras.layers.Layer):
    def __init__(self,
                 state_dim,
                 T,
                 aggr_method='mean',
                 attn_method='sym-gat',
                 update_method='gru',
                 attn_head=2,
                 activation='relu'):
        """
        A Massage passing neural network that takes edge pairs.

        Args:
            state_dim: int, the hidden dimension of node embeddings
            T: int, the number of messaging passing
            aggr_method: str, the message aggregation function
            attn_method: str, the node attention function
            update_method: str, the update function
            attn_head: int, the number of attention heads
            activation: str, the activation function
        """
        super(SPARSE_MPNN, self).__init__(self)
        self.state_dim = state_dim
        self.T = T
        self.activation = activations.get(activation)
        self.aggr_method = aggr_method
        self.attn_method = attn_method
        self.attn_head = attn_head
        self.update_method = update_method

    def build(self, input_shape):
        self.embed = tf.keras.layers.Dense(self.state_dim, activation=self.activation)
        self.MP = MP_layer(self.state_dim, self.aggr_method, self.activation,
                           self.attn_method, self.attn_head, self.update_method)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: list, a list of input tensors
                X: an input node embedding tensor [?, N, F]
                A: an input edge pair tensor [?, P, 2]
                E: an input edge embedding tensor [?, P, S]
                mask: an input node embedding mask tensor [?, N, F]
                degree: an input edge pair gcn attention tensor 1/sqrt(N(i)*N(j)) [?, P]

        Returns:
            X: an output node embedding tensor [?, N, F']
        """
        X, A, E, mask, degree = inputs
        A = tf.cast(A, tf.int32)
        X = self.embed(X)
        for _ in range(self.T):
            X = self.MP([X, A, E, mask, degree])
        return X


class MP_layer(tf.keras.layers.Layer):
    def __init__(self, state_dim, aggr_method, activation, attn_method, attn_head, update_method):
        super(MP_layer, self).__init__(self)
        self.state_dim = state_dim
        self.aggr_method = aggr_method
        self.activation = activation
        self.attn_method = attn_method
        self.attn_head = attn_head
        self.update_method = update_method

    def build(self, input_shape):
        self.message_passer = Message_Passer_NNM(self.state_dim, self.attn_head, self.attn_method,
                                                 self.aggr_method, self.activation)
        if self.update_method == 'gru':
            self.update_functions = Update_Func_GRU(self.state_dim)
        elif self.update_method == 'mlp':
            self.update_functions = Update_Func_MLP(self.state_dim, self.activation)

        self.built = True

    def call(self, inputs, **kwargs):
        X, A, E, mask, degree = inputs  # [?,N,C],[?,E,2],[?,E,S],[?,1],[?,1]
        agg_m = self.message_passer([X, A, E, degree])  # [?,N,C]
        mask = tf.tile(mask[..., None], [1, 1, self.state_dim])
        agg_m = tf.multiply(agg_m, mask)
        updated_nodes = self.update_functions([X, agg_m])  # [?,N,C]
        updated_nodes = tf.multiply(updated_nodes, mask)
        return updated_nodes


class Message_Passer_NNM(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads, attn_method, aggr_method, activation):
        super(Message_Passer_NNM, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads
        self.attn_method = attn_method
        self.aggr_method = aggr_method
        self.activation = activation


    def build(self, input_shape):
        self.nn = tf.keras.layers.Dense(units=self.state_dim * self.state_dim * self.attn_heads,
                                        activation=self.activation)

        if self.attn_method == 'gat':
            self.attn_func = Attention_GAT(self.state_dim, self.attn_heads)
        elif self.attn_method == 'sym-gat':
            self.attn_func = Attention_SYM_GAT(self.state_dim, self.attn_heads)
        elif self.attn_method == 'cos':
            self.attn_func = Attention_COS(self.state_dim, self.attn_heads)
        elif self.attn_method == 'linear':
            self.attn_func = Attention_Linear(self.state_dim, self.attn_heads)
        elif self.attn_method == 'gen-linear':
            self.attn_func = Attention_Gen_Linear(self.state_dim, self.attn_heads)
        elif self.attn_method == 'const':
            self.attn_func = Attention_Const(self.state_dim, self.attn_heads)
        elif self.attn_method == 'gcn':
            self.attn_func = Attention_GCN(self.state_dim, self.attn_heads)

        self.bias = self.add_weight(name='attn_bias', shape=[self.state_dim], initializer='zeros')
        self.built = True

    def call(self, inputs, **kwargs):
        # Edge network
        X, A, E, degree = inputs
        N = K.int_shape(X)[1]
        targets, sources = A[..., -2], A[..., -1]
        W = self.nn(E)
        W = tf.reshape(W, [-1, tf.shape(E)[1], self.attn_heads, self.state_dim, self.state_dim])
        X = tf.tile(X[..., None], [1, 1, 1, self.attn_heads])
        X = tf.transpose(X, [0, 1, 3, 2])

        # Attention
        attn_coef = self.attn_func([X, N, targets, sources, degree])
        messages = tf.gather(X, sources, batch_dims=1)
        messages = messages[..., None]
        messages = tf.matmul(W, messages)
        messages = messages[..., 0]
        output = attn_coef * messages
        num_rows = tf.shape(targets)[0]
        rows_idx = tf.range(num_rows)
        segment_ids_per_row = targets + N * tf.expand_dims(rows_idx, axis=1)


        # Aggregation
        if self.aggr_method == 'max':
            output = tf.math.unsorted_segment_max(output, segment_ids_per_row, N * num_rows)
        elif self.aggr_method == 'mean':
            output = tf.math.unsorted_segment_mean(output, segment_ids_per_row, N * num_rows)
        elif self.aggr_method == 'sum':
            output = tf.math.unsorted_segment_sum(output, segment_ids_per_row, N * num_rows)

        # Output
        output = tf.reshape(output, [-1, N, self.attn_heads, self.state_dim])
        output = tf.reduce_mean(output, axis=-2)
        output = K.bias_add(output, self.bias)
        return output


class Update_Func_GRU(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.state_dim = state_dim

    def build(self, input_shape):
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(self.state_dim)
        self.built = True

    def call(self, inputs, **kwargs):
        # Remember node dim
        old_state, agg_messages = inputs
        B, N, F = K.int_shape(old_state)
        B1, N1, F1 = K.int_shape(agg_messages)
        # Reshape so GRU can be applied, concat so old_state and messages are in sequence
        old_state = tf.reshape(old_state, [-1, 1, F])
        agg_messages = tf.reshape(agg_messages, [-1, 1, F1])
        concat = self.concat_layer([old_state, agg_messages])
        # Apply GRU and then reshape so it can be returned
        activation = self.GRU(concat)
        activation = tf.reshape(activation, [-1, N, F])
        return activation


class Update_Func_MLP(tf.keras.layers.Layer):
    def __init__(self, state_dim, activation):
        super(Update_Func_MLP, self).__init__()
        self.state_dim = state_dim
        self.activation = activation

    def build(self, input_shape):
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(self.state_dim, activation=self.activation)

    def call(self, inputs, **kwargs):
        # Remember node dim
        old_state, agg_messages = inputs
        # Reshape so GRU can be applied, concat so old_state and messages are in sequence
        concat = self.concat_layer([old_state, agg_messages])
        # Apply GRU and then reshape so it can be returned
        activation = self.dense(concat)
        return activation


class Attention_GAT(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_GAT, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(name='attn_kernel_self', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(name='attn_kernel_adjc', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = attn_for_self + attn_for_adjc
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(attn_coef - tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N), targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_SYM_GAT(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_SYM_GAT, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(name='attn_kernel_self', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(name='attn_kernel_adjc', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)

        attn_for_self_reverse = tf.gather(attn_for_self, sources, batch_dims=1)
        attn_for_adjc_reverse = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_coef = attn_for_self + attn_for_adjc + attn_for_self_reverse + attn_for_adjc_reverse
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(attn_coef - tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N), targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_COS(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_COS, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(name='attn_kernel_self', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(name='attn_kernel_adjc', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = tf.multiply(attn_for_self, attn_for_adjc)
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(attn_coef - tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N), targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_Linear(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_Linear, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_adjc = self.add_weight(name='attn_kernel_adjc', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        X, N, targets, sources, _ = inputs
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = attn_for_adjc
        attn_coef = tf.nn.tanh(attn_coef)
        attn_coef = tf.exp(attn_coef - tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N), targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_Gen_Linear(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_Gen_Linear, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(name='attn_kernel_self', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(name='attn_kernel_adjc', shape=[self.state_dim, self.attn_heads, 1],
                                                initializer='glorot_uniform')
        self.gen_nn = tf.keras.layers.Dense(units=self.attn_heads,
                                            kernel_initializer='glorot_uniform', use_bias=False)
        self.built = True

    def call(self, inputs, **kwargs):
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = attn_for_self + attn_for_adjc
        attn_coef = tf.nn.tanh(attn_coef)
        attn_coef = self.gen_nn(attn_coef)
        attn_coef = tf.exp(attn_coef - tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N), targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef, targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_GCN(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_GCN, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def call(self, inputs, **kwargs):
        _, _, _, _, degree = inputs
        attn_coef = degree[..., None, None]
        attn_coef = tf.tile(attn_coef, [1, 1, self.attn_heads, 1])
        return attn_coef


class Attention_Const(tf.keras.layers.Layer):
    def __init__(self, state_dim, attn_heads):
        super(Attention_Const, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def call(self, inputs, **kwargs):
        _, _, targets, _, degree = inputs
        attn_coef = tf.ones((tf.shape(targets)[0], tf.shape(targets)[1], self.attn_heads, 1))
        return attn_coef


class GlobalAttentionPool(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):
        super(GlobalAttentionPool, self).__init__()
        self.state_dim = state_dim
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAttentionPool"

    def build(self, input_shape):
        self.features_layer = Dense(self.state_dim, name='features_layer')
        self.attention_layer = Dense(self.state_dim, name='attention_layer', activation='sigmoid')
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_linear = self.features_layer(inputs)
        attn = self.attention_layer(inputs)
        masked_inputs = inputs_linear * attn
        output = K.sum(masked_inputs, axis=-2, keepdims=False)
        return output


class GlobalAttentionSumPool(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalAttentionSumPool, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GraphAttentionSumPool"

    def build(self, input_shape):
        F = int(input_shape[-1])
        # Attention kernels
        self.attn_kernel = self.add_weight(shape=(F, 1),
                                           initializer='glorot_uniform',
                                           name='attn_kernel')
        self.built = True

    def call(self, inputs, **kwargs):
        X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        output = K.batch_dot(attn_coeff, X)
        return output
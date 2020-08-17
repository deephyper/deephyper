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


class Apply1DMask(tf.keras.layers.Layer):
    def __init__(self):
        super(Apply1DMask, self).__init__()
        pass

    def __str__(self):
        return f"Apply a 1D mask to the dense layers"

    def call(self, inputs, **kwargs):
        assert type(inputs) is list
        return tf.multiply(inputs[0], inputs[1])


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
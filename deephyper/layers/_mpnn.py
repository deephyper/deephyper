import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense


def get_gcn_attention(edge_pair):
    r"""calculate gcn attention coef for an edge pair.
    Args:
        edge_pair (array): # edges * 2
    """
    # The inverse of sqrt of node degrees
    source_node = np.unique(edge_pair[:, 0], return_counts=True, return_inverse=True)
    target_node = np.unique(edge_pair[:, 1], return_counts=True, return_inverse=True)
    source_neighbor = []
    for element in source_node[1]:
        source_neighbor.append(source_node[2][element])
    target_neighbor = []
    for element in target_node[1]:
        target_neighbor.append(target_node[2][element])
    # the number of neighboring nodes of each node in each edge pair
    source_neighbor = np.array(source_neighbor)
    target_neighbor = np.array(target_neighbor)

    neighbor_multiplication = np.multiply(source_neighbor, target_neighbor)
    gcn_attention = 1 / np.sqrt(neighbor_multiplication)
    return gcn_attention


def get_mol_feature(mol, max_node, max_edge):
    r"""get molecular features for an rdkit molecule
    Args:
        mol (rdkit molecule)
        max_node (int): maximum number of nodes in a dataset
        max_edge (int): maximum number of edges in a dataset
    """
    node_feat = mol.nodes
    edge_pair = mol.pair_edges.T
    edge_feat = mol.pairs

    num_node = node_feat.shape[0]
    num_node_feat = node_feat.shape[1]
    num_edge = edge_feat.shape[0]
    num_edge_feat = edge_feat.shape[1]

    node_feat_new = np.zeros(shape=(max_node, num_node_feat))
    node_feat_new[:num_node, :num_node_feat] = node_feat

    edge_pair_new = np.ones(shape=(max_edge + max_node, 2)) * (max_node - 1)
    edge_pair_new[:num_edge, :] = edge_pair

    edge_pair_new[max_edge:, 0] = np.arange(max_node)
    edge_pair_new[max_edge:, 1] = np.arange(max_node)

    edge_feat_new = np.zeros(shape=(max_edge + max_node, num_edge_feat))
    edge_feat_new[:num_edge, :] = edge_feat

    mask = np.zeros(shape=(max_node,))
    mask[:num_node] = 1

    gcn_attention = get_gcn_attention(edge_pair_new)
    return node_feat_new, edge_pair_new, edge_feat_new, mask, gcn_attention


def get_all_mol_feat(data, max_node, max_edge):
    r"""get molecular features for a whole dataset
    Args:
        data (weave data)
        max_node (int): maximum number of nodes in a dataset
        max_edge (int): maximum number of edges in a dataset
    """
    x = data.X
    y = data.y

    node_feat, edge_pair, edge_feat, mask, gcn_attention = [], [], [], [], []

    for mol in x:
        node_feat_temp, edge_pair_temp, edge_feat_temp, mask_temp, gcn_attention_temp = get_mol_feature(mol, max_node,
                                                                                                        max_edge)
        node_feat.append(node_feat_temp)
        edge_pair.append(edge_pair_temp)
        edge_feat.append(edge_feat_temp)
        mask.append(mask_temp)
        gcn_attention.append(gcn_attention_temp)

    if len(y.shape) == 1:
        y = y[..., None]

    return [np.array(node_feat), np.array(edge_pair), np.array(edge_feat), np.array(mask),
            np.array(gcn_attention)], y


class SPARSE_MPNN(tf.keras.layers.Layer):
    """Message passing cell.

    Args:
        state_dim (int): number of output channels.
        T (int): number of message passing repetition.
        attn_heads (int): number of attention heads.
        attn_method (str): type of attention methods.
        aggr_method (str): type of aggregation methods.
        activation (str): type of activation functions.
        update_method (str): type of update functions.
    """

    def __init__(self,
                 state_dim,
                 T,
                 aggr_method,
                 attn_method,
                 update_method,
                 attn_head,
                 activation):
        super(SPARSE_MPNN, self).__init__(self)
        self.state_dim = state_dim
        self.T = T
        self.activation = activations.get(activation)
        self.aggr_method = aggr_method
        self.attn_method = attn_method
        self.attn_head = attn_head
        self.update_method = update_method

    def build(self, input_shape):
        self.embed = tf.keras.layers.Dense(
            self.state_dim, activation=self.activation)
        self.MP = MP_layer(
            self.state_dim,
            self.aggr_method,
            self.activation,
            self.attn_method,
            self.attn_head,
            self.update_method)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor (batch size * # nodes * # node features)
                A (tensor): edge pair tensor (batch size * # edges * 2), one is source ID, one is target ID
                E (tensor): edge feature tensor (batch size * # edges * # edge features)
                mask (tensor): node mask tensor to mask out non-existent nodes (batch size * # nodes)
                degree (tensor): node degree tensor for GCN attention (batch size * # edges)

        Returns: X (tensor): results after several repetitions of edge network, attention, aggregation and update
        function (batch size * # nodes * # node features)
        """
        # the input contains a list of five tensors
        X, A, E, mask, degree = inputs
        # edge pair needs to be in the int format
        A = tf.cast(A, tf.int32)
        # this is a limitation of MPNN in general, the node feature is mapped to (batch size * # nodes * # node
        # features)
        X = self.embed(X)
        # run T times message passing
        for _ in range(self.T):
            X = self.MP([X, A, E, mask, degree])
        return X


class MP_layer(tf.keras.layers.Layer):
    """Message passing layer.

    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
        attn_method (str): type of attention methods.
        aggr_method (str): type of aggregation methods.
        activation (str): type of activation functions.
        update_method (str): type of update functions.
    """

    def __init__(
            self,
            state_dim,
            aggr_method,
            activation,
            attn_method,
            attn_head,
            update_method):
        super(MP_layer, self).__init__(self)
        self.state_dim = state_dim
        self.aggr_method = aggr_method
        self.activation = activation
        self.attn_method = attn_method
        self.attn_head = attn_head
        self.update_method = update_method

    def build(self, input_shape):
        self.message_passer = Message_Passer_NNM(
            self.state_dim,
            self.attn_head,
            self.attn_method,
            self.aggr_method,
            self.activation)
        if self.update_method == 'gru':
            self.update_functions = Update_Func_GRU(self.state_dim)
        elif self.update_method == 'mlp':
            self.update_functions = Update_Func_MLP(
                self.state_dim, self.activation)

        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor (batch size * # nodes * state dimension)
                A (tensor): edge pair tensor (batch size * # edges * 2), one is source ID, one is target ID
                E (tensor): edge feature tensor (batch size * # edges * # edge features)
                mask (tensor): node mask tensor to mask out non-existent nodes (batch size * # nodes)
                degree (tensor): node degree tensor for GCN attention (batch size * # edges)

        Returns: updated_nodes (tensor): results after edge network, attention, aggregation and update function
        (batch size * # nodes * state dimension)
        """
        # the input contains a list of five tensors
        X, A, E, mask, degree = inputs
        # use the message passing to generate aggregated results
        # agg_m (batch size * # nodes * state dimension)
        agg_m = self.message_passer([X, A, E, degree])
        # expand the mask to (batch size * # nodes * state dimension)
        mask = tf.tile(mask[..., None], [1, 1, self.state_dim])
        # use the mask to screen out non-existent nodes
        # agg_m (batch size * # nodes * state dimension)
        agg_m = tf.multiply(agg_m, mask)
        # update function using the old node feature X and new aggregated node feature agg_m
        # updated_nodes (batch size * # nodes * state dimension)
        updated_nodes = self.update_functions([X, agg_m])
        # use the mask to screen out non-existent nodes
        # updated_nodes (batch size * # nodes * state dimension)
        updated_nodes = tf.multiply(updated_nodes, mask)
        return updated_nodes


class Message_Passer_NNM(tf.keras.layers.Layer):
    """Message passing kernel.

    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
        attn_method (str): type of attention methods.
        aggr_method (str): type of aggregation methods.
        activation (str): type of activation functions.
    """

    def __init__(
            self,
            state_dim,
            attn_heads,
            attn_method,
            aggr_method,
            activation):
        super(Message_Passer_NNM, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads
        self.attn_method = attn_method
        self.aggr_method = aggr_method
        self.activation = activation

    def build(self, input_shape):
        self.nn1 = tf.keras.layers.Dense(units=32,
                                         activation=tf.nn.relu)

        self.nn2 = tf.keras.layers.Dense(units=32,
                                         activation=tf.nn.relu)

        self.nn3 = tf.keras.layers.Dense(units=self.attn_heads * self.state_dim * self.state_dim,
                                         activation=tf.nn.relu)

        if self.attn_method == 'gat':
            self.attn_func = Attention_GAT(self.state_dim, self.attn_heads)
        elif self.attn_method == 'sym-gat':
            self.attn_func = Attention_SYM_GAT(self.state_dim, self.attn_heads)
        elif self.attn_method == 'cos':
            self.attn_func = Attention_COS(self.state_dim, self.attn_heads)
        elif self.attn_method == 'linear':
            self.attn_func = Attention_Linear(self.state_dim, self.attn_heads)
        elif self.attn_method == 'gen-linear':
            self.attn_func = Attention_Gen_Linear(
                self.state_dim, self.attn_heads)
        elif self.attn_method == 'const':
            self.attn_func = Attention_Const(self.state_dim, self.attn_heads)
        elif self.attn_method == 'gcn':
            self.attn_func = Attention_GCN(self.state_dim, self.attn_heads)

        self.bias = self.add_weight(
            name='attn_bias', shape=[
                self.state_dim], initializer='zeros')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor (batch size * # nodes * state dimension)
                A (tensor): edge pair tensor (batch size * # edges * 2), one is source ID, one is target ID
                E (tensor): edge feature tensor (batch size * # edges * # edge features)
                degree (tensor): node degree tensor for GCN attention (batch size * # edges)

        Returns: output (tensor): results after edge network, attention and aggregation
        (batch size * # nodes * state dimension)
        """
        # Edge network to transform edge information to message weight
        # the input contains a list of four tensors
        X, A, E, degree = inputs
        # N is the number of nodes (scalar)
        N = K.int_shape(X)[1]
        # extract target and source IDs from the edge pair
        # targets (batch size * # edges)
        # sources (batch size * # edges)
        targets, sources = A[..., -2], A[..., -1]
        # the first edge network layer that maps edge features to a weight tensor W
        # W (batch size * # edges * 128)
        W = self.nn1(E)
        # W (batch size * # edges * 128)
        W = self.nn2(W)
        # W (batch size * # edges * state dimension ** 2)
        W = self.nn3(W)
        # reshape W to (batch size * # edges * #attention heads * state dimension * state dimension)
        W = tf.reshape(W, [-1, tf.shape(E)[1], self.attn_heads, self.state_dim, self.state_dim])
        # expand the dimension of node features to
        # (batch size * # nodes * state dimension * #attention heads)
        X = tf.tile(X[..., None], [1, 1, 1, self.attn_heads])
        # transpose the node features to
        # (batch size * # nodes * #attention heads * node features)
        X = tf.transpose(X, [0, 1, 3, 2])
        # attention added to the message weight
        # attn_coef (batch size * # edges * #attention heads * state dimension)
        attn_coef = self.attn_func([X, N, targets, sources, degree])
        # gather source node features
        # The batch_dims argument lets you gather different items from each element of a batch.
        # Using batch_dims=1 is equivalent to having an outer loop over the first axis of params and indices:
        # Here is an example from https://www.tensorflow.org/api_docs/python/tf/gather
        # params = tf.constant([
        #     [0, 0, 1, 0, 2],
        #     [3, 0, 0, 0, 4],
        #     [0, 5, 0, 6, 0]])
        # indices = tf.constant([
        #     [2, 4],
        #     [0, 4],
        #     [1, 3]])
        # tf.gather(params, indices, axis=1, batch_dims=1).numpy()
        # array([[1, 2],
        #        [3, 4],
        #        [5, 6]], dtype=int32)
        # messages (batch size * # edges * #attention heads * state dimension)
        messages = tf.gather(X, sources, batch_dims=1, axis=1)
        # messages (batch size * # edges * #attention heads * state dimension * 1)
        messages = messages[..., None]
        # W (batch size *  # edges * #attention heads * state dimension * state dimension)
        # messages (batch size * # edges * #attention heads * state dimension * 1)
        # --> messages (batch size * # edges * #attention heads * state dimension * 1)
        messages = tf.matmul(W, messages)
        # messages (batch size * # edges * #attention heads * state dimension)
        messages = messages[..., 0]
        # attn_coef (batch size * # edges * # attention heads * state dimension)
        # messages (batch size * # edges * # attention heads * state dimension)
        # --> output (batch size *  # edges * # attention heads * state dimension)
        output = attn_coef * messages
        # batch size
        num_rows = tf.shape(targets)[0]
        # [0, ..., batch size] (batch size)
        rows_idx = tf.range(num_rows)
        # N is # nodes, add this to distinguish each batch
        segment_ids_per_row = targets + N * tf.expand_dims(rows_idx, axis=1)
        # Aggregation to summarize neighboring node messages
        # output (batch size *  # nodes * # attention heads * state dimension)
        if self.aggr_method == 'max':
            output = tf.math.unsorted_segment_max(
                output, segment_ids_per_row, N * num_rows)
        elif self.aggr_method == 'mean':
            output = tf.math.unsorted_segment_mean(
                output, segment_ids_per_row, N * num_rows)
        elif self.aggr_method == 'sum':
            output = tf.math.unsorted_segment_sum(
                output, segment_ids_per_row, N * num_rows)
        # output the mean of all attention heads
        # output (batch size * # nodes * # attention heads * state dimension)
        output = tf.reshape(output, [-1, N, self.attn_heads, self.state_dim])
        # output (batch size * # nodes * state dimension)
        output = tf.reduce_mean(output, axis=-2)
        # add bias, output (batch size * # nodes * state dimension)
        output = K.bias_add(output, self.bias)
        return output


class Update_Func_GRU(tf.keras.layers.Layer):
    """Gated recurrent unit update function. Check details here https://arxiv.org/abs/1412.3555

    Args:
        state_dim (int): number of output channels.
    """

    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.state_dim = state_dim

    def build(self, input_shape):
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(self.state_dim)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                old_state (tensor): node hidden feature tensor (batch size * # nodes * state dimension)
                agg_messages (tensor): node hidden feature tensor (batch size * # nodes * state dimension)

        Returns:
            activation (tensor): activated tensor from update function (batch size * # nodes * state dimension)
        """
        # Remember node dim
        # old_state (batch size * # nodes * state dimension)
        # agg_messages (batch size * # nodes * state dimension)
        old_state, agg_messages = inputs
        # B is batch size
        # N is # nodes
        # F is # node features = state dimension
        B, N, F = K.int_shape(old_state)
        # similar to B, N, F
        B1, N1, F1 = K.int_shape(agg_messages)
        # reshape so GRU can be applied, concat so old_state and messages are in sequence
        # old_state (batch size * # nodes * 1 * state dimension)
        old_state = tf.reshape(old_state, [-1, 1, F])
        # agg_messages (batch size * # nodes * 1 * state dimension)
        agg_messages = tf.reshape(agg_messages, [-1, 1, F1])
        # agg_messages (batch size * # nodes * 2 * state dimension)
        concat = self.concat_layer([old_state, agg_messages])
        # Apply GRU and then reshape so it can be returned
        # activation (batch size * # nodes * state dimension)
        activation = self.GRU(concat)
        activation = tf.reshape(activation, [-1, N, F])
        return activation


class Update_Func_MLP(tf.keras.layers.Layer):
    """Multi-layer perceptron update function.

    Args:
        state_dim (int): number of output channels.
        activation (str): the type of activation functions.
    """

    def __init__(self, state_dim, activation):
        super(Update_Func_MLP, self).__init__()
        self.state_dim = state_dim
        self.activation = activation

    def build(self, input_shape):
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(
            self.state_dim,
            activation=self.activation,
            kernel_initializer='zeros')

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                old_state (tensor): node hidden feature tensor
                agg_messages (tensor): node hidden feature tensor

        Returns:
            activation (tensor): activated tensor from update function (
        """
        old_state, agg_messages = inputs
        concat = self.concat_layer([old_state, agg_messages])
        activation = self.dense(concat)
        return activation


class Attention_GAT(tf.keras.layers.Layer):
    """GAT Attention. Check details here https://arxiv.org/abs/1710.10903

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \text{LeakyReLU}(\textbf{a}(\textbf{Wh}_i||\textbf{Wh}_j))
        $$
        where \(\textbf{a}\) is a trainable vector, and \(||\) represents concatenation.
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_GAT, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(
            name='attn_kernel_adjc',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = attn_for_self + attn_for_adjc
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(
            attn_coef -
            tf.gather(
                tf.math.unsorted_segment_max(
                    attn_coef,
                    targets,
                    N),
                targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef,
                                                            targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_SYM_GAT(tf.keras.layers.Layer):
    """GAT Symmetry Attention.

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \alpha_{ij} + \alpha_{ij}
        $$
        based on GAT.
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_SYM_GAT, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(
            name='attn_kernel_adjc',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)

        attn_for_self_reverse = tf.gather(attn_for_self, sources, batch_dims=1)
        attn_for_adjc_reverse = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_coef = attn_for_self + attn_for_adjc + \
                    attn_for_self_reverse + attn_for_adjc_reverse
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(
            attn_coef -
            tf.gather(
                tf.math.unsorted_segment_max(
                    attn_coef,
                    targets,
                    N),
                targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef,
                                                            targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_COS(tf.keras.layers.Layer):
    """COS Attention. Check details here https://arxiv.org/abs/1803.07294

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \textbf{a}(\textbf{Wh}_i || \textbf{Wh}_j)
        $$
        where \(\textbf{a}\) is a trainable vector.
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_COS, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(
            name='attn_kernel_adjc',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor (batch, E, H, 1)
        """
        X, N, targets, sources, _ = inputs
        attn_kernel_self = tf.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_self = tf.reduce_sum(X * attn_kernel_self[None, ...], -1)
        attn_for_self = tf.gather(attn_for_self, targets, batch_dims=1)
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = tf.multiply(attn_for_self, attn_for_adjc)
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.exp(
            attn_coef -
            tf.gather(
                tf.math.unsorted_segment_max(
                    attn_coef,
                    targets,
                    N),
                targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef,
                                                            targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_Linear(tf.keras.layers.Layer):
    """Linear Attention.

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \text{tanh} (\textbf{a}_l\textbf{Wh}_i + \textbf{a}_r\textbf{Wh}_j)
        $$
        where \(\textbf{a}_l\) and \(\textbf{a}_r\) are trainable vectors.
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_Linear, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_adjc = self.add_weight(
            name='attn_kernel_adjc',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
        X, N, targets, sources, _ = inputs
        attn_kernel_adjc = tf.transpose(self.attn_kernel_adjc, (2, 1, 0))
        attn_for_adjc = tf.reduce_sum(X * attn_kernel_adjc[None, ...], -1)
        attn_for_adjc = tf.gather(attn_for_adjc, sources, batch_dims=1)
        attn_coef = attn_for_adjc
        attn_coef = tf.nn.tanh(attn_coef)
        attn_coef = tf.exp(
            attn_coef -
            tf.gather(
                tf.math.unsorted_segment_max(
                    attn_coef,
                    targets,
                    N),
                targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef,
                                                            targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_Gen_Linear(tf.keras.layers.Layer):
    """Generalized Linear Attention. Check details here https://arxiv.org/abs/1802.00910

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \textbf{W}_G \text{tanh} (\textbf{Wh}_i + \textbf{Wh}_j)
        $$
        where \(\textbf{W}_G\) is a trainable matrix.
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_Gen_Linear, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.attn_kernel_adjc = self.add_weight(
            name='attn_kernel_adjc',
            shape=[
                self.state_dim,
                self.attn_heads,
                1],
            initializer='glorot_uniform')
        self.gen_nn = tf.keras.layers.Dense(
            units=self.attn_heads,
            kernel_initializer='glorot_uniform',
            use_bias=False)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
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
        attn_coef = tf.exp(
            attn_coef -
            tf.gather(
                tf.math.unsorted_segment_max(
                    attn_coef,
                    targets,
                    N),
                targets))
        attn_coef /= tf.gather(tf.math.unsorted_segment_max(attn_coef,
                                                            targets, N) + 1e-9, targets)
        attn_coef = tf.nn.dropout(attn_coef, 0.5)
        attn_coef = attn_coef[..., None]
        return attn_coef


class Attention_GCN(tf.keras.layers.Layer):
    """GCN Attention.

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}}
        $$
        where \(\mathcal{N}(i)\) is the number of neighboring nodes of node \(i\).
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_GCN, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
        _, _, _, _, degree = inputs
        attn_coef = degree[..., None, None]
        attn_coef = tf.tile(attn_coef, [1, 1, self.attn_heads, 1])
        return attn_coef


class Attention_Const(tf.keras.layers.Layer):
    """Constant Attention.

        The attention coefficient between node $i$ and $j$ is calculated as:
        $$
            \alpha_{ij} = 1.
        $$
    Args:
        state_dim (int): number of output channels.
        attn_heads (int): number of attention heads.
    """

    def __init__(self, state_dim, attn_heads):
        super(Attention_Const, self).__init__()
        self.state_dim = state_dim
        self.attn_heads = attn_heads

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (list):
                X (tensor): node feature tensor
                N (int): number of nodes
                targets (tensor): target node index tensor
                sources (tensor): source node index tensor
                degree (tensor): node degree sqrt tensor (for GCN attention)

        Returns:
            attn_coef (tensor): attention coefficient tensor
        """
        _, _, targets, _, degree = inputs
        attn_coef = tf.ones(
            (tf.shape(targets)[0],
             tf.shape(targets)[1],
             self.attn_heads,
             1))
        return attn_coef


class GlobalAttentionPool(tf.keras.layers.Layer):
    """Global Attention Pool.

        A gated attention global pooling layer as presented by [Li et al. (2017)](https://arxiv.org/abs/1511.05493).
        Details can be seen from https://github.com/danielegrattarola/spektral
    Args:
        state_dim (int): number of output channels.
    """

    def __init__(self, state_dim, **kwargs):
        super(GlobalAttentionPool, self).__init__()
        self.state_dim = state_dim
        self.kwargs = kwargs

    def __str__(self):
        return f"GlobalAttentionPool"

    def build(self, input_shape):
        self.features_layer = Dense(self.state_dim, name='features_layer')
        self.attention_layer = Dense(
            self.state_dim,
            name='attention_layer',
            activation='sigmoid')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): the node feature tensor

        Returns:
            GlobalAttentionPool tensor (tensor)
        """
        inputs_linear = self.features_layer(inputs)
        attn = self.attention_layer(inputs)
        masked_inputs = inputs_linear * attn
        output = K.sum(masked_inputs, axis=-2, keepdims=False)
        return output


class GlobalAttentionSumPool(tf.keras.layers.Layer):
    """Global Attention Summation Pool.

        Pools a graph by learning attention coefficients to sum node features.
        Details can be seen from https://github.com/danielegrattarola/spektral
    """

    def __init__(self, **kwargs):
        super(GlobalAttentionSumPool, self).__init__()
        self.kwargs = kwargs

    def __str__(self):
        return f"GlobalAttentionSumPool"

    def build(self, input_shape):
        F = int(input_shape[-1])
        # Attention kernels
        self.attn_kernel = self.add_weight(shape=(F, 1),
                                           initializer='glorot_uniform',
                                           name='attn_kernel')
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): the node feature tensor

        Returns:
            GlobalAttentionSumPool tensor (tensor)
        """
        X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        output = K.batch_dot(attn_coeff, X)
        return output


class GlobalAvgPool(tf.keras.layers.Layer):
    """Global Average Pool.

        Takes the average over all the nodes or features.
        Details can be seen from https://github.com/danielegrattarola/spektral
    Args:
        axis (int): the axis to take average.
    """

    def __init__(self, axis=-2, **kwargs):
        super(GlobalAvgPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GlobalAvgPool"

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): the node feature tensor

        Returns:
            GlobalAvgPool tensor (tensor)
        """
        return tf.reduce_mean(inputs, axis=self.axis)


class GlobalMaxPool(tf.keras.layers.Layer):
    """Global Max Pool.

        Takes the max value over all the nodes or features.
        Details can be seen from https://github.com/danielegrattarola/spektral
    Args:
        axis (int): the axis to take the max value.
    """

    def __init__(self, axis=-2, **kwargs):
        super(GlobalMaxPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GlobalMaxPool"

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): the node feature tensor

        Returns:
            GlobalMaxPool tensor (tensor)
        """
        return tf.reduce_max(inputs, axis=self.axis)


class GlobalSumPool(tf.keras.layers.Layer):
    """Global Summation Pool.

        Takes the summation over all the nodes or features.
        Details can be seen from https://github.com/danielegrattarola/spektral

    Args:
        axis (int): the axis to take summation.
    """

    def __init__(self, axis=-2, **kwargs):
        super(GlobalSumPool, self).__init__()
        self.axis = axis
        self.kwargs = kwargs

    def __str__(self):
        return f"GlobalSumPool"

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (tensor): the node feature tensor

        Returns:
            GlobalSumPool tensor (tensor)
        """
        return tf.reduce_sum(inputs, axis=self.axis)

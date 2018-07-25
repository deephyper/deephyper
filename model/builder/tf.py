'''
 * @Author: romain.egele, dipendra jha
 * @Date: 2018-06-21 09:11:29
'''

import tensorflow as tf

import deephyper.model.arch as a
from deephyper.model.utilities.train_utils import *
from deephyper.search import util

logger = util.conf_logger('deephyper.model.builder.tf')

class BasicBuilder:
    def __init__(self, config, arch_def):
        self.hyper_params = config[a.hyperparameters]
        self.learning_rate = config[a.hyperparameters][a.learning_rate]
        self.optimizer_name = config[a.hyperparameters][a.optimizer]
        self.batch_size = config[a.hyperparameters][a.batch_size]
        self.loss_metric_name = config[a.hyperparameters][a.loss_metric]
        self.test_metric_name = config[a.hyperparameters][a.test_metric]
        self.train_size = config['train_size']
        self.input_shape = config[a.input_shape] #for image it is [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], for vector [NUM_ATTRIBUTES]
        self.num_outputs = config[a.num_outputs]
        self.is_classifier = self.num_outputs > 1
        self.arch_def = arch_def
        self.conv1D_params = { a.num_filters: 32,
                               a.filter_size: 3,
                               a.stride_size: 1,
                               a.pool_size: 1,
                               a.drop_out: 1,
                               a.padding: 'SAME',
                               a.activation: a.relu,
                               a.batch_norm: False,
                               a.batch_norm_bef: True}
        self.conv2D_params = { a.num_filters: 32,
                               a.filter_height: 5,
                               a.filter_width: 5,
                               a.stride_height: 1,
                               a.stride_width: 1,
                               a.pool_height: 2,
                               a.pool_width: 2,
                               a.padding: 'SAME',
                               a.activation: a.relu,
                               a.batch_norm: False,
                               a.batch_norm_bef: True,
                               a.drop_out: 0.8}

        self.dense_params = { a.num_outputs: 1024,
                              a.drop_out: 1,
                              a.batch_norm: False,
                              a.batch_norm_bef: True,
                              a.activation: a.relu}
        self.tempconv_params = {a.num_filters: 32,
                              a.filter_size: 3,
                              a.stride_size: 1,
                              a.pool_size: 1,
                              a.drop_out: 1,
                              a.padding: 'SAME',
                              a.dilation: 3,
                              a.activation: a.relu,
                              a.batch_norm: False,
                              a.batch_norm_bef: True}
        self.rnn_params = {
            a.num_units:64,
            a.unit_type: 'LSTM',
            a.drop_out: 1
        }

        self.act_dict = {a.relu: tf.nn.relu,
                         a.sigmoid: tf.nn.sigmoid, a.tanh: tf.nn.tanh}
        self.layer_types_default_value = {
            a.conv1D: self.conv1D_params,
            a.conv2D: self.conv2D_params,
            a.dense: self.dense_params,
            a.tempconv: self.tempconv_params,
            a.rnn: self.rnn_params,
            a.tempconv: self.tempconv_params
        }
        self.define_model()

    def set_default(self, cfg_layer):
        layer_type = cfg_layer.get(a.layer_type)
        assert layer_type != None
        cfg_default = self.layer_types_default_value[layer_type]

        for k in cfg_default:
            if cfg_layer.get(k) == layer_type:
                continue
            if cfg_layer.get(k) == None:
                cfg_layer[k] = cfg_default[k]

    def define_model(self):
        self.tf_label_type = np.float32 if self.num_outputs == 1 else np.int64
        self.train_data_node = tf.placeholder(tf.float32,
            shape=([self.batch_size] + self.input_shape))
        self.train_labels_node = tf.placeholder(self.tf_label_type,
                                                shape=([self.batch_size]))
        #self.train_labels_node = tf.placeholder(self.tf_label_type,
        #    shape=([self.batch_size, self.num_outputs]))
        self.eval_data_node = tf.placeholder(tf.float32,
            shape=([self.batch_size] + self.input_shape))
        self.logits = self.build_graph(self.train_data_node)
        self.logits = tf.squeeze(self.logits)
        self.eval_preds = self.build_graph(self.eval_data_node, train=False)
        self.eval_preds = tf.squeeze(self.eval_preds)
        self.loss_metric = selectLossMetric(self.loss_metric_name)
        self.test_metric = selectTestMetric(self.test_metric_name)
        self.loss = self.loss_metric(self.train_labels_node, self.logits)
        #if 'mean' not in self.loss_metric_name:
        #    self.loss = tf.reduce_mean(self.loss)
        self.batch = tf.Variable(0)
        self.optimizer_fn = selectOptimizer(self.optimizer_name)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.batch*self.batch_size, self.train_size, 0.95, staircase=True)
        self.optimizer = self.optimizer_fn(learning_rate).minimize(self.loss)

    def get_layer_input(self, nets, skip_conns, last_layer = False):
        if not skip_conns and not last_layer: return nets[0]
        self.used_nets |= set(skip_conns)
        if not last_layer:
            inps = [nets[i] for i in skip_conns]
        else:
            inps = [nets[i] for i in range(len(nets)) if i not in self.used_nets]
        input_layer = inps[0]
        for i in range(1, len(inps)):
            curr_layer_shape = input_layer.get_shape().as_list()
            next_layer_shape = inps[i].get_shape().as_list()
            assert len(curr_layer_shape) == len(next_layer_shape), 'Concatenation of two tensors with different dimensions not supported.'
            max_shape = [ max(curr_layer_shape[x], next_layer_shape[x]) for x in range(len(curr_layer_shape))]
            curr_layer_padding_len = [[0,0]]
            next_layer_padding_len = [[0,0]]
            for d in range(1, len(max_shape[1:-1])+1):
                curr_layer_padding_len.append([
                    (max_shape[d] - curr_layer_shape[d]) // 2,
                    (max_shape[d] - curr_layer_shape[d]) -
                    ((max_shape[d] - curr_layer_shape[d]) // 2)])
                next_layer_padding_len.append(
                    [(max_shape[d] - next_layer_shape[d]) // 2,
                     (max_shape[d] - next_layer_shape[d]) -
                     ((max_shape[d] - next_layer_shape[d]) // 2)])
            curr_layer_padding_len.append([0,0])
            next_layer_padding_len.append([0,0])
            if sum([sum(x) for x in curr_layer_padding_len]) != 0:
                input_layer = tf.pad(input_layer, curr_layer_padding_len, 'CONSTANT')
            next_layer = inps[i]
            if sum([sum(x) for x in next_layer_padding_len]) != 0:
                next_layer = tf.pad(next_layer, next_layer_padding_len, 'CONSTANT')
            input_layer = tf.concat([input_layer, next_layer], len(max_shape)-1)
        return input_layer

    def build_graph(self, data_node, train=True):
        net = data_node
        nets = [net]
        reuse = None if train else True
        weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
        arch_keys = ['layer_'+str(i) for i in range(1,len(self.arch_def)+1)]
        self.used_nets = set()
        for arch_key in arch_keys:
            with tf.name_scope(arch_key):
                layer_params = self.arch_def[arch_key]
                self.set_default(layer_params)
                layer_type = layer_params[a.layer_type]
                assert layer_type in [a.conv1D, a.conv2D, a.dense, a.tempconv]
                activation = self.act_dict[layer_params[a.activation]] if layer_params[
                    a.activation] in self.act_dict else tf.nn.relu
                if 'skip_conn' in layer_params:
                    net = self.get_layer_input(nets, layer_params['skip_conn'])
                else:
                    net = nets[-1]
                print (net.get_shape(), type(net.get_shape()), list(net.get_shape()))
                if layer_type == a.conv2D:
                    conv_params = self.conv2D_params.copy()
                    conv_params.update(layer_params)
                    num_filters = conv_params[a.num_filters]
                    filter_width = conv_params[a.filter_width]
                    filter_height = conv_params[a.filter_height]
                    padding = conv_params[a.padding]
                    stride_height = conv_params[a.stride_height]
                    stride_width = conv_params[a.stride_width]
                    pool_width = conv_params[a.pool_width]
                    pool_height = conv_params[a.pool_height]
                    if conv_params[a.batch_norm]:
                        if conv_params[a.batch_norm_bef]:
                            net = tf.layers.conv2d(net, filters=num_filters, kernel_size=[filter_height, filter_width], strides=[
                                                   stride_height, stride_width], padding=padding, kernel_initializer=weights_initializer,activation=None, reuse=reuse, name=arch_key+'/{0}'.format(a.conv2D))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key+'/{0}'.format(a.batch_norm))
                            net = activation(net)
                        else:
                            net = tf.layers.conv2d(net,
                                                   filters=num_filters,
                                                   kernel_size=[filter_height,
                                                                filter_width],
                                                   strides=[stride_height, stride_width], padding=padding,
                                                   kernel_initializer=weights_initializer,activation=activation,
                                                   reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.conv2D))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key + '/{0}'.format(a.batch_norm))
                    else:
                        net = tf.layers.conv2d(net, filters=num_filters, kernel_size=[filter_height, filter_width],
                                               strides=[
                                                   stride_height, stride_width], padding=padding,
                                               kernel_initializer=weights_initializer, activation=activation, reuse=reuse,
                                               name=arch_key + '/{0}'.format(a.conv2D))
                    if pool_height != 1 and pool_width !=1:
                        net = tf.layers.max_pooling2d(net, [pool_height, pool_width], strides=[1, 1])
                elif layer_type == a.conv1D:
                    conv_params = layer_params
                    num_filters = conv_params[a.num_filters]
                    filter_size = conv_params[a.filter_size]
                    padding = conv_params[a.padding]
                    stride_size = conv_params[a.stride_size]
                    pool_size = conv_params[a.pool_size]
                    if conv_params[a.batch_norm]:
                        if conv_params[a.batch_norm_bef]:
                            net = tf.layers.conv1d(net,
                                                   filters=num_filters,
                                                   kernel_size=[filter_size],
                                                   strides=[stride_size],
                                                   padding=padding,
                                                   kernel_initializer=weights_initializer,
                                                   activation=None,
                                                   reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.conv1D))
                            net = tf.layers.batch_normalization(
                                    net,
                                    reuse=reuse,
                                    name=arch_key + '/{0}'.format(a.batch_norm))
                            net = activation(net)
                        else:
                            net = tf.layers.conv1d(net, filters=num_filters, kernel_size=[filter_size],
                                                   strides=[
                                                       stride_size], padding=padding,
                                                   kernel_initializer=weights_initializer, activation=activation,
                                                   reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.conv1D))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key + '/{0}'.format(a.batch_norm))
                    else:
                        net = tf.layers.conv1d(net,
                                               filters=num_filters,
                                               kernel_size=[filter_size],
                                               strides=[stride_size],
                                               padding=padding,
                                               kernel_initializer=weights_initializer, activation=activation,
                                               reuse=reuse,
                                               name=arch_key + '/{0}'.format(a.conv1D))
                    if pool_size !=1:
                        net = tf.layers.max_pooling1d(net, pool_size, strides=1)
                elif layer_type == a.tempconv:
                    conv_params = layer_params
                    num_filters = conv_params[a.num_filters]
                    filter_size = conv_params[a.filter_size]
                    stride_size = conv_params[a.stride_size]
                    dilation = (conv_params[a.dilation])
                    padding_len = (filter_size-1)*dilation
                    pad_arr = tf.constant([(0, 0), (padding_len, 0), (0, 0)])
                    net = tf.pad(net, pad_arr)
                    pool_size = conv_params[a.pool_size]

                    dilation = (dilation,)
                    padding = 'VALID'
                    if conv_params[a.batch_norm]:
                        if conv_params[a.batch_norm_bef]:
                            net = tf.layers.conv1d(net,
                                                   filters=num_filters,
                                                   kernel_size=[filter_size],
                                                   strides=[stride_size],
                                                   dilation_rate=dilation,
                                                   padding=padding,
                                                   kernel_initializer=weights_initializer,
                                                   activation=None,
                                                   data_format='channels_last',
                                                   reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.tempconv))
                            net = tf.layers.batch_normalization(
                                net,
                                reuse=reuse,
                                name=arch_key + '/{0}'.format(a.batch_norm))
                            net = activation(net)
                        else:
                            net = tf.layers.conv1d(net, filters=num_filters, kernel_size=[filter_size],
                                                   strides=[
                                                       stride_size], padding=padding, dilation_rate = dilation,
                                                   kernel_initializer=weights_initializer,
                                                   data_format='channels_last',
                                                   activation=activation,
                                                   reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.tempconv))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key + '/{0}'.format(a.batch_norm))
                    else:
                        net = tf.layers.conv1d(net,
                                               filters=num_filters,
                                               kernel_size=[filter_size],
                                               strides=[stride_size],
                                               padding=padding,
                                               dilation_rate=dilation,
                                               kernel_initializer=weights_initializer, activation=activation,
                                               reuse=reuse,
                                               data_format='channels_last',
                                               name=arch_key + '/{0}'.format(a.tempconv))
                    net = tf.contrib.layers.layer_norm(net)
                    if pool_size != 1:
                        net = tf.layers.max_pooling1d(net, pool_size, strides=1)
                elif layer_type == a.dense:
                    if len(net.get_shape()> 2):
                        net = tf.contrib.layers.flatten(net)
                    dense_params = self.dense_params.copy()
                    dense_params.update(layer_params)
                    num_outputs = dense_params[a.num_outputs]
                    if dense_params[a.batch_norm]:
                        if dense_params[a.batch_norm_bef]:
                            net = tf.layers.dense(
                                net, units=num_outputs, kernel_initializer=weights_initializer, name=arch_key+'/{0}'.format(a.dense), reuse=reuse, activation=None)
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key+'/{0}'.format(a.batch_norm))
                            net = activation(net)
                        else:
                            net = tf.layers.dense(net, units=num_outputs, kernel_initializer=weights_initializer,
                                                  name=arch_key + '/{0}'.format(a.dense), reuse=reuse, activation=activation)
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key + '/{0}'.format(a.batch_norm))
                    else:
                        net = tf.layers.dense(net, units=num_outputs, kernel_initializer=weights_initializer,
                                              name=arch_key + '/{0}'.format(a.dense), reuse=reuse, activation=activation)

            dropout = layer_params[a.drop_out]
            if dropout < 1.0:
                net = tf.nn.dropout(net, keep_prob=dropout)
            nets.append(net)
        if 'skip_conns' in layer_params:
            net = self.get_layer_input(nets, skip_conns=[], last_layer=True)
        else:
            net = nets[-1]
        net = tf.contrib.layers.flatten(net)
        #net = tf.layers.dense(net, units=512, name='hereItIs', reuse=reuse ) # TODO !!!
        #net = tf.nn.dropout(net, keep_prob=0.8)
        net = tf.layers.dense(net, units=self.num_outputs, name='output_layer',reuse=reuse)
        nets.append(net)
        return net


class RNNModel:
    def __init__(self, config, arch_def):
        self.hyper_params = config[a.hyperparameters]
        self.learning_rate = config[a.hyperparameters][a.learning_rate]
        self.optimizer_name = config[a.hyperparameters][a.optimizer]
        self.batch_size = config[a.hyperparameters][a.batch_size]
        self.loss_metric_name = config[a.hyperparameters][a.loss_metric]
        self.test_metric_name = config[a.hyperparameters][a.test_metric]
        self.num_steps = config[a.num_steps] #for image it is [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], for vector [NUM_ATTRIBUTES]
        self.num_outputs = config[a.num_outputs]
        self.num_features = config[a.num_features]
        self.text_input = config[a.text_input]
        self.regression = config[a.regression]
        self.eval_batch_size = config[a.hyperparameters][a.eval_batch_size]
        self.unit_type = config[a.unit_type]
        self.arch_def = arch_def
        self.vocab_size = len(config[a.data][a.vocabulary])
        self.conv1D_params = { a.num_filters: 32,
                               a.filter_size: 3,
                               a.stride_size: 1,
                               a.pool_size: 1,
                               a.drop_out: 1,
                               a.padding: 'SAME',
                               a.activation: a.relu,
                               a.batch_norm: False,
                               a.batch_norm_bef: True}
        self.conv2D_params = { a.num_filters: 32,
                               a.filter_height: 3,
                               a.filter_width: 3,
                               a.stride_height: 1,
                               a.stride_width: 1,
                               a.pool_height: 2,
                               a.pool_width: 2,
                               a.padding: 'SAME',
                               a.activation: a.relu,
                               a.batch_norm: False,
                               a.batch_norm_bef: True,
                               a.drop_out: 1}

        self.dense_params = { a.num_outputs: 1024,
                              a.drop_out: 1,
                              a.batch_norm: False,
                              a.batch_norm_bef: True,
                              a.activation: a.relu}
        self.tempconv_params = {a.num_filters: 32,
                              a.filter_size: 3,
                              a.stride_size: 1,
                              a.pool_size: 1,
                              a.drop_out: 1,
                              a.padding: 'SAME',
                              a.dilation: 3,
                              a.activation: a.relu,
                              a.batch_norm: False,
                              a.batch_norm_bef: True}
        self.rnn_params = {
            a.num_units:64,
            a.unit_type: 'LSTM',
            a.drop_out: 1
        }

        self.act_dict = {a.relu: tf.nn.relu,
                         a.sigmoid: tf.nn.sigmoid, a.tanh: tf.nn.tanh}
        self.layer_types_default_value = {
            a.conv1D: self.conv1D_params,
            a.conv2D: self.conv2D_params,
            a.dense: self.dense_params,
            a.tempconv: self.tempconv_params,
            a.rnn: self.rnn_params
        }
        self.define_model()

    def set_default(self, cfg_layer):
        layer_type = cfg_layer.get(a.layer_type)
        assert layer_type != None
        cfg_default = self.layer_types_default_value[layer_type]

        for k in cfg_default:
            if cfg_layer.get(k) == layer_type:
                continue
            if cfg_layer.get(k) == None:
                cfg_layer[k] = cfg_default[k]

    def define_model(self):
        if self.text_input:
            self.train_data_node = tf.placeholder(tf.int64, [self.batch_size, self.num_steps])
            self.train_labels_node = tf.placeholder(tf.int64, [self.batch_size, self.num_steps])
            self.eval_data_node = tf.placeholder(tf.int64, [self.eval_batch_size, self.num_steps])
            self.eval_labels_node = tf.placeholder(tf.int64, [self.eval_batch_size, self.num_steps])

        else:
            self.train_data_node = tf.placeholder(tf.int64, [self.batch_size, self.num_steps, self.num_features])
            self.train_labels_node = tf.placeholder(tf.int64, [self.batch_size, self.num_steps])
            self.eval_data_node = tf.placeholder(tf.int64, [self.eval_batch_size, self.num_steps, self.num_features])
            self.eval_labels_node = tf.placeholder(tf.int64, [self.eval_batch_size, self.num_steps])
        logger.debug(f'Train data size: {self.train_data_node.get_shape()}, {self.batch_size}, {self.num_steps}, {self.num_features}')
        logger.debug('Building training graph')
        logits, final_state = self.build_graph(self.train_data_node, train=True)
        self.logits = logits
        self.final_state = final_state
        logger.debug('Building evaluation graph')
        self.eval_preds, _= self.build_graph(self.eval_data_node, train=False)
        self.loss_metric = selectLossMetric(self.loss_metric_name)
        if self.test_metric_name == 'perplexity':
            self.test_metric = lambda x: np.exp
        else: self.test_metric = selectTestMetric(self.test_metric_name)

        if self.loss_metric_name == 'sequence_loss_by_example':
            loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
            self.loss = loss_fun([self.logits], [tf.reshape(self.train_labels_node, [-1])],
                            [tf.ones([self.batch_size * self.num_steps])],
                            self.vocab_size)
            self.loss = tf.reduce_sum(self.loss) / (self.batch_size * self.num_steps)
            self.eval_loss = loss_fun([self.eval_preds], [tf.reshape(self.eval_labels_node, [-1])],
                                 [tf.ones([self.eval_batch_size * self.num_steps])],
                                 self.vocab_size)
            self.eval_loss = tf.reduce_sum(self.eval_loss) / (self.eval_batch_size * self.num_steps)
        else:
            self.loss = self.loss_metric(self.train_labels_node, self.logits)
            self.eval_loss = self.loss_metric(self.eval_labels_node, self.eval_preds)
            if 'mean' not in self.loss_metric_name:
                self.loss = tf.reduce_mean(self.loss)
        self.batch = tf.Variable(0)
        self.optimizer_fn = selectOptimizer(self.optimizer_name)
        self.optimizer = self.optimizer_fn(self.learning_rate).minimize(self.loss)
        logger.debug('done with defining model')

    def build_graph(self, input_data, train=True):
        reuse = None if train else True
        if self.text_input:
            embed_size = self.arch_def['layer_1']['num_units']
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
                embedding = tf.get_variable(
                'embedding', [self.vocab_size, embed_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_data)
        else:
            inputs = input_data

        arch_keys = ['layer_' + str(i) for i in range(1,len(self.arch_def)+1)]
        rnn_layers = []
        last_num_units = 0
        for arch_key in arch_keys:
            layer_params = self.arch_def[arch_key]
            last_num_units = layer_params['num_units']
            self.set_default(layer_params)
            if self.unit_type == 'LSTM':
                layer = tf.contrib.rnn.BasicLSTMCell(layer_params['num_units'], forget_bias=0.0, state_is_tuple=True, reuse=reuse)
            elif self.unit_type == 'GRU':
                layer = tf.contrib.rnn.GRUCell(layer_params['num_units'], forget_bias=0.0, reuse=reuse)
            if train and layer_params['drop_out'] < 1:
                layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=layer_params['drop_out'])
            rnn_layers.append(layer)
        logger.debug('creating multicellRNN')
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        self.initial_state = state = stacked_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        logger.debug('inputs shape {}'.format(inputs.get_shape()))

        outputs, state = tf.nn.dynamic_rnn(
             stacked_lstm, inputs, initial_state=self.initial_state, dtype=tf.float32, time_major=False)
        logger.debug('cell outputs shape {}'.format(outputs.get_shape()))
        last_num_units = int(last_num_units)

        if self.regression:
            logger.debug('Last num units is {}'.format(last_num_units))
            output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, last_num_units])
            #output = tf.reshape(outputs, [-1, last_num_units])

            logger.debug('cell output shape after reshaping:{}'.format(output.get_shape()))
            # Logits and output
            logits = tf.contrib.layers.fully_connected(output, self.vocab_size, activation_fn=tf.nn.softmax)
            logger.debug('logit output shape:{}'.format(logits.get_shape()))

        else:
            output = tf.reshape(tf.concat(axis=1, values=outputs), [self.batch_size,])
            #output = tf.reshape(outputs, [self.batch_size, ])
            logger.debug('cell output shape after reshaping: {}'.format(output.get_shape()))
            # Logits and output
            logits = tf.contrib.layers.fully_connected(output, self.num_outputs, activation_fn=tf.nn.softmax)
            logger.debug('logit output shape: {}'.format(logits.get_shape()))
        return logits, state

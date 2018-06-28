'''
 * @Author: romain.egele, dipendra jha
 * @Date: 2018-06-21 09:11:29
'''

import tensorflow as tf

import deephyper.model.arch as a
from deephyper.model.utilities.train_utils import *


class BasicBuilder:
    def __init__(self, config, arch_def):
        self.hyper_params = config[a.hyperparameters]
        self.learning_rate = config[a.hyperparameters][a.learning_rate]
        self.optimizer_name = config[a.hyperparameters][a.optimizer]
        self.batch_size = config[a.hyperparameters][a.batch_size]
        self.loss_metric_name = config[a.hyperparameters][a.loss_metric]
        self.test_metric_name = config[a.hyperparameters][a.test_metric]
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
        self.act_dict = {a.relu: tf.nn.relu,
                         a.sigmoid: tf.nn.sigmoid, a.tanh: tf.nn.tanh}
        self.layer_types_default_value = {
            a.conv1D: self.conv1D_params,
            a.conv2D: self.conv2D_params,
            a.dense: self.dense_params
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
        if 'mean' not in self.loss_metric_name:
            self.loss = tf.reduce_mean(self.loss)
        self.batch = tf.Variable(0)
        self.optimizer_fn = selectOptimizer(self.optimizer_name)
        self.optimizer = self.optimizer_fn(self.learning_rate).minimize(self.loss)

    def build_graph(self, data_node, train=True):
        net = data_node
        nets = [net]
        reuse = None if train else True
        weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
        arch_keys = ['layer_'+str(i) for i in range(len(self.arch_def))]
        for arch_key in arch_keys:
            with tf.name_scope(arch_key):
                layer_params = self.arch_def[arch_key]
                self.set_default(layer_params)
                layer_type = layer_params[a.layer_type]
                assert layer_type in [a.conv1D, a.conv2D, a.dense]
                activation = self.act_dict[layer_params[a.activation]] if layer_params[
                    a.activation] in self.act_dict else tf.nn.relu
                if layer_type == a.conv2D:
                    conv_params = self.conv2D_params.copy()
                    conv_params.update(layer_params)
                    num_filters = conv_params[a.num_filters]
                    filter_width = conv_params[a.filter_width]
                    filter_height = conv_params[a.filter_height]
                    padding = conv_params[a.padding]
                    stride_height = conv_params[a.stride_height]
                    stride_width = conv_params[a.stride_width]
                    if conv_params[a.batch_norm]:
                        if conv_params[a.batch_norm_bef]:
                            net = tf.layers.conv2d(net, filters=num_filters, kernel_size=[filter_height, filter_width], strides=[
                                                   stride_height, stride_width], padding=padding, kernel_initializer=weights_initializer, activation=None, reuse=reuse, name=arch_key+'/{0}'.format(a.conv2D))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key+'/{0}'.format(a.batch_norm))
                            net = activation(net)
                        else:
                            net = tf.layers.conv2d(net, filters=num_filters, kernel_size=[filter_height, filter_width],
                                                   strides=[
                                                       stride_height, stride_width], padding=padding,
                                                   kernel_initializer=weights_initializer, activation=activation, reuse=reuse,
                                                   name=arch_key + '/{0}'.format(a.conv2D))
                            net = tf.layers.batch_normalization(
                                net, reuse=reuse, name=arch_key + '/{0}'.format(a.batch_norm))
                    else:
                        net = tf.layers.conv2d(net, filters=num_filters, kernel_size=[filter_height, filter_width],
                                               strides=[
                                                   stride_height, stride_width], padding=padding,
                                               kernel_initializer=weights_initializer, activation=activation, reuse=reuse,
                                               name=arch_key + '/{0}'.format(a.conv2D))
                elif layer_type == a.conv1D:
                    conv_params = layer_params
                    num_filters = conv_params[a.num_filters]
                    filter_size = conv_params[a.filter_size]
                    padding = conv_params[a.padding]
                    stride_size = conv_params[a.stride_size]
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
                elif layer_type == a.dense:
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
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, units=self.num_outputs, name='output_layer',reuse=reuse)
        nets.append(net)
        return net

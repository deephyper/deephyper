import numpy as np
import tensorflow as tf

import deephyper.searches.nas.model.arch as a
from deephyper.searches import util
from deephyper.searches.nas.model.train_utils import *

logger = util.conf_logger('deephyper.searches.nas.model.builder')

class BasicBuilder:
    def __init__(self, config, arch_def):
        self.hyper_params = config[a.hyperparameters]
        self.learning_rate = config[a.hyperparameters][a.learning_rate]
        self.optimizer_name = config[a.hyperparameters][a.optimizer]
        self.batch_size = config[a.hyperparameters][a.batch_size]
        self.loss_metric_name = config[a.hyperparameters][a.loss_metric]
        self.test_metric_name = config[a.hyperparameters][a.test_metric]
        self.train_size = config['train_size']
        self.regression = config['regression']
        self.input_shape = config[a.input_shape] #for image it is [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], for vector [NUM_ATTRIBUTES]
        self.output_shape = config[a.output_shape]
        self.num_outputs = config[a.output_shape][0]
        logger.debug(f'input_shape: {self.input_shape}')
        logger.debug(f'output_shape: {self.output_shape}')
        self.arch_def = arch_def

        self.create_structure_func = config['create_structure']['func']
        self.create_structure_kwargs = config['create_structure']['kwargs']

        self.define_model()

    def define_model(self):
        logger.debug('Defining model')

        self.tf_label_type = np.float32

        self.train_data_node = tf.placeholder(tf.float32,
            shape=([None] + self.input_shape))
        self.train_labels_node = tf.placeholder(self.tf_label_type,
                                                shape=([None] + self.output_shape))

        self.eval_data_node = tf.placeholder(tf.float32,
            shape=([None] + self.input_shape))
        logger.debug(f'self.eval_data_node: {self.eval_data_node.get_shape()}')

        logger.debug('Building training graph')
        self.logits = self.build_graph(self.train_data_node, train=True)

        logger.debug('Building evaluation graph')
        self.eval_preds = self.build_graph(self.eval_data_node, train=False)

        logger.debug('Defining optimizers')
        self.loss_metric = selectLossMetric(self.loss_metric_name)
        self.test_metric = selectTestMetric(self.test_metric_name)
        self.loss = self.loss_metric(self.train_labels_node, self.logits)

        if not self.regression:
            self.eval_preds = tf.nn.softmax(self.eval_preds)
            self.logits = tf.nn.softmax(self.logits)

        self.batch = tf.Variable(0)
        self.optimizer_fn = selectOptimizer(self.optimizer_name)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
            self.batch * self.batch_size,
            self.train_size,
            0.95,
            staircase=True)
        self.optimizer = self.optimizer_fn(learning_rate).minimize(self.loss)
        logger.debug('Done defining model')


    def build_graph(self, data_node, train=True):
        # final_activation = None if self.regression else tf.nn.sigmoid
        final_activation = None

        input_placeholder = data_node
        network = self.create_structure_func(input_placeholder,
                                             **self.create_structure_kwargs)
        network.set_ops(self.arch_def)
        if train:
            network.draw_graphviz('graph.dot')

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            merge_tensor = network.create_tensor(train=train)

            net = tf.contrib.layers.flatten(merge_tensor)
            net = tf.layers.dense(net, units=self.num_outputs, name='output_layer',
                kernel_initializer=tf.initializers.random_uniform(),
                activation=final_activation)

        return net

    def get_number_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

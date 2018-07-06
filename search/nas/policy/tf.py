'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 10:23:16
'''
import tensorflow as tf


class NASCellPolicy:
    def __init__(self, num_features):
        '''
            * num_feature : number of features you want to search per layer
        '''
        self.num_features = num_features

    def get(self, state, max_layers):
        with tf.name_scope('policy_network'):
            nas_cell = tf.contrib.rnn.NASCell(num_units=self.num_features*max_layers)

            outputs, state = tf.nn.dynamic_rnn(
                cell=nas_cell,
                inputs=tf.expand_dims(state, -1),
                dtype=tf.float32
                )

            bias = tf.Variable([0.05]*self.num_features*max_layers)
            outputs = tf.nn.bias_add(outputs, bias)

            # Returned last output of rnn
            return outputs[:, -1:, :]

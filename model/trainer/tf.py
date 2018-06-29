'''
 * @Author: romain.egele
 * @Date: 2018-06-19 13:04:50

  Basic training class for models, using tensorflow.

'''
import math
import time

import numpy as np
import tensorflow as tf

import deephyper.model.arch as a
from deephyper.model.builder.tf import BasicBuilder


class BasicTrainer:
    '''
      BasicTrainer class aims to train models
    '''

    def __init__(self, config):
        self.config = config
        self.config_hp = config[a.hyperparameters]
        # self.config_summary = config[a.summary]
        # self.config_logs = config[a.logs]
        self.input_shape = config[a.input_shape]
        self.num_outputs = config[a.num_outputs]
        self.data = config[a.data]
        self.batch_size = self.config_hp[a.batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]
        self.train_X = None
        self.train_y = None
        self.valid_X = None
        self.valid_y = None
        self.train_size = None
        self.preprocess_data()
        self.patience = self.config_hp[a.patience] if a.patience in self.config_hp else int(
            self.train_size/self.batch_size * self.num_epochs/5)
        self.eval_freq = self.config_hp[a.eval_freq] if a.eval_freq in self.config_hp else self.train_size/self.batch_size

    def preprocess_data(self):
        self.train_X = self.config[a.data][a.train_X]
        self.train_y = self.config[a.data][a.train_Y]
        perm = np.random.permutation(np.shape(self.train_X)[0])
        self.train_X = self.train_X[0][perm]
        self.train_y = self.train_y[0][perm]
        self.valid_X = self.config[a.data][a.valid_X]
        self.valid_y = self.config[a.data][a.valid_Y]
        self.train_size = np.shape(self.config[a.data][a.train_X])[0]
        print(self.train_X.shape, self.train_y.shape, self.input_shape)
        self.train_X = self.train_X.reshape(
            [-1]+self.input_shape).astype('float32')
        self.valid_X = self.valid_X[0].reshape(
            [-1]+self.input_shape).astype('float32')
        self.np_label_type = 'float32' if self.num_outputs == 1 else 'int64'
        self.train_y = np.squeeze(self.train_y).astype(self.np_label_type)
        self.valid_y = np.squeeze(self.valid_y).astype(self.np_label_type)

    def eval_in_batches(self, model, data, sess):
        size = data.shape[0]
        if size < self.batch_size:
            raise ValueError(
                'batch size for evals larger than dataset: %d' % size)
        if self.num_outputs > 1:
            predictions = np.ndarray(
                shape=(size, self.num_outputs), dtype=np.float32)
            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                if end <= size:
                    predictions[begin:end, :] = sess.run(model.eval_preds, feed_dict={
                                                         model.eval_data_node: data[begin:end, ...]})
                else:
                    batch_predictions = sess.run(model.eval_preds, feed_dict={
                                                 model.eval_data_node: data[-self.batch_size:, ...]})
                    predictions[-self.batch_size:, :] = batch_predictions
        else:
            predictions = np.ndarray(shape=(size), dtype=np.float32)
            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                if end <= size:
                    predictions[begin:end] = sess.run(model.eval_preds,
                                                      feed_dict={model.eval_data_node: data[begin:end, ...]})
                else:
                    batch_predictions = sess.run(model.eval_preds,
                                                 feed_dict={model.eval_data_node: data[-self.batch_size:, ...]})
                    predictions[-self.batch_size:] = batch_predictions
        return predictions

    def get_rewards(self, architecture, global_step=''):
        tf.reset_default_graph()
        best_step = 0
        best_res = 0 if self.num_outputs > 1 else float('inf')
        metric_term = 'error' if self.num_outputs ==1 else 'accuracy'
        put_perc = '%' if self.num_outputs > 1 else ''
        with tf.Graph().as_default() as g:
            model = BasicBuilder(self.config, architecture)
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                if (self.config.get(a.summary)):
                    self.summary_writer = tf.summary.FileWriter(
                        'logs/run_{0}'.format(global_step), graph=tf.get_default_graph())
                start_time = time.time()
                for step in range((self.num_epochs * self.train_size) // self.batch_size):
                    offset = (
                        step * self.batch_size) % (self.train_size - self.batch_size)
                    batch_data = self.train_X[offset:(
                        offset + self.batch_size), ...]
                    batch_labels = self.train_y[offset:(
                        offset + self.batch_size),]
                    feed_dict = {model.train_data_node: batch_data,
                                 model.train_labels_node: batch_labels}
                    _, l, predictions = sess.run([model.optimizer, model.loss,model.logits],
                                                     feed_dict=feed_dict)
                    if step % self.eval_freq == 0:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) % self.batch_size / self.train_size,
                                                                 1000 * elapsed_time / self.eval_freq),
                                                                 end=', ')
                        print('Minibatch loss: %.3f' % (l), end=', ')
                        if put_perc:
                            print('Minibatch %s: %.3f%%' %(metric_term,
                              model.test_metric(predictions, batch_labels)), end=', ')
                        else:
                            print('Minibatch %s: %.3f%%' % (metric_term,
                                                           model.test_metric(predictions, batch_labels)), end=', ')
                        valid_preds = self.eval_in_batches(
                            model, self.valid_X, sess)
                        valid_res = model.test_metric(valid_preds, self.valid_y)
                        if put_perc:
                            print('Validation %s: %.3f%%' %(metric_term, valid_res))
                        else:
                            print('Validation %s: %.3f%%' %(metric_term, valid_res))

                        if self.num_outputs > 1 and best_res < valid_res:
                            best_res = valid_res
                            best_step = step
                        elif self.num_outputs == 1 and best_res > valid_res:
                            best_res = valid_res
                            best_step = step
                    if best_step + self.patience < step:
                        break
        return best_res

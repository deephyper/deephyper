'''
 * @Author: romain.egele, dipendra jha
 * @Date: 2018-06-19 13:04:50

  Basic training class for models, using tensorflow.

'''
import math
import time

import numpy as np
import tensorflow as tf

import deephyper.model.arch as a
from deephyper.search import util
from deephyper.model.builder.tf import BasicBuilder, RNNModel

logger = util.conf_logger('deephyper.model.trainer.tf')

class BasicTrainer:
    '''
      BasicTrainer class aims to train models
    '''

    def __init__(self, config):
        logger.debug('[PARAM] Instantiate BasicTrainer')
        self.config = config
        self.config_hp = config[a.hyperparameters]
        # self.config_summary = config[a.summary]
        # self.config_logs = config[a.logs]
        self.input_shape = config[a.input_shape]
        self.num_outputs = config[a.num_outputs]
        if config[a.layer_type] == a.rnn:
            self.num_steps = config[a.num_steps]
            self.vocab_size = self.config[a.vocab_size]

        self.regression = config[a.regression]
        self.data = config[a.data]
        self.batch_size = self.config_hp[a.batch_size]
        self.eval_batch_size = self.config_hp[a.eval_batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]
        self.train_X = None
        self.train_y = None
        self.valid_X = None
        self.valid_y = None
        self.train_size = None
        self.preprocess_data()
        self.patience = self.config_hp[a.patience] if a.patience in self.config_hp else int(
            self.train_size/self.batch_size * self.num_epochs/5.)
        self.eval_freq = self.config_hp[a.eval_freq] if a.eval_freq in self.config_hp else self.train_size//self.batch_size
        logger.debug('[PARAM] BasicTrainer instantiated')

    def preprocess_data(self):
        self.train_X = self.config[a.data][a.train_X]
        self.train_y = self.config[a.data][a.train_Y]
        perm = np.random.permutation(np.shape(self.train_X)[0])
        self.train_X = self.train_X[perm]
        self.train_y = self.train_y[perm]
        self.valid_X = self.config[a.data][a.valid_X]
        self.valid_y = self.config[a.data][a.valid_Y]
        self.train_size = np.shape(self.config[a.data][a.train_X])[0]
        #if self.train_size == self.batch_size: self.train_size = self.train_X.shape[1]
        logger.debug(f'\ntrain_X.shape = {self.train_X.shape},\n\
                       train_y.shape = {self.train_y.shape},\n\
                       input_shape = {self.input_shape}')
        self.train_X = self.train_X.reshape(
            [-1]+self.input_shape).astype('float32')
        self.valid_X = self.valid_X.reshape(
            [-1]+self.input_shape).astype('float32')
        self.np_label_type = 'float32' if self.num_outputs == 1 else 'int64'
        self.train_y = np.squeeze(self.train_y).astype(self.np_label_type)
        self.valid_y = np.squeeze(self.valid_y).astype(self.np_label_type)

        logger.debug(f'\n after reshaping: train_X.shape = {self.train_X.shape},\n\
                           train_y.shape = {self.train_y.shape},\n\
                           input_shape = {self.input_shape}')

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

    def eval_rnn_in_batches(self, model, data, labels, sess):
        size = data.shape[0]
        val_res = 0
        if size < self.batch_size:
            raise ValueError(
                'batch size for evals larger than dataset: %d' % size)
        if self.num_outputs > 1:
            predictions = np.ndarray(
                shape=(size, self.num_outputs), dtype=np.float32)
            for begin in range(0, size, self.eval_batch_size):
                end = begin + self.batch_size
                if end <= size:
                    predictions[begin:end, :], curr_loss = sess.run([model.eval_preds, model.eval_loss], feed_dict={
                                                         model.eval_data_node: data[begin:end, ...], model.eval_labels_node: labels[begin:end,...]})
                else:
                    batch_predictions, curr_loss = sess.run([model.eval_preds, model.eval_loss], feed_dict={
                                                 model.eval_data_node: data[-self.batch_size:, ...], model.eval_labels_node: labels[begin:end,...]})
                    predictions[-self.batch_size:, :] = batch_predictions
                val_res += curr_loss
        else:
            predictions = np.ndarray(shape=(size*self.num_steps, self.vocab_size), dtype=np.float32)
            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                if end <= size:
                    predictions[begin*self.num_steps:end*self.num_steps,:], curr_loss = sess.run([model.eval_preds,model.eval_loss],
                                                      feed_dict={model.eval_data_node: data[begin:end, ...], model.eval_labels_node: labels[begin:end,...]})
                else:
                    batch_predictions, curr_loss = sess.run([model.eval_preds,model.eval_loss],
                                                 feed_dict={model.eval_data_node: data[-self.batch_size:, ...], model.eval_labels_node: labels[-self.eval_batch_size:,...]})
                    predictions[-self.batch_size*self.num_steps:,] = batch_predictions
                val_res += curr_loss
        return predictions, val_res/(size/self.eval_batch_size)


    def get_rewards(self, architecture, global_step=''):
        if self.config[a.layer_type] == 'rnn':
            return self.get_rewards_from_rnn(architecture, global_step='')
        else:
            return self.get_rewards_default(architecture, global_step='')

    def get_rewards_from_rnn(self, architecture, global_step=''):
        tf.reset_default_graph()

        best_step = 0
        best_res = 0 if self.num_outputs > 1 else float('inf')
        metric_term = 'error' if self.num_outputs ==1 else 'accuracy'
        put_perc = '%' if self.num_outputs > 1 else ''
        with tf.Graph().as_default() as g:
            model = RNNModel(self.config, architecture)
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                if (self.config.get(a.summary)):
                    self.summary_writer = tf.summary.FileWriter(
                        'logs/run_{0}'.format(global_step), graph=tf.get_default_graph())
                start_time = time.time()
                logger.debug('Train size is : ', self.train_size)
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
                    if step % (self.eval_freq // 10 if self.eval_freq // 10 else 1) == 0:

                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        logs = f'Step {step} (epoch {float(step) * self.batch_size / self.train_size}), {1000 * elapsed_time / self.eval_freq} ms, '
                        logs += f'Minibatch loss: {l}, '
                        if model.test_metric_name == 'perplexity':
                            train_res = np.exp(l)
                        else:
                            train_res = model.test_metric(predictions, batch_labels)
                        if put_perc:
                            logs += 'Minibatch %s: %.3f%%, ' %(metric_term,
                              train_res)
                        else:
                            logs += 'Minibatch %s: %.3f, ' % (metric_term,
                                                           train_res)
                        logger.debug(logs)

                    if step % self.eval_freq == 0:

                        valid_preds, valid_loss = self.eval_rnn_in_batches(
                            model, self.valid_X, self.valid_y, sess)
                        if model.test_metric_name == 'perplexity':
                            valid_res = np.exp(valid_loss)
                        else:
                            valid_res = model.test_metric(predictions, batch_labels)

                        if put_perc:
                            logger.debug('Validation %s: %.3f%%' %(metric_term, valid_res))
                        else:
                            logger.debug('Validation %s: %.3f' %(metric_term, valid_res))

                        if self.num_outputs > 1 and best_res < valid_res:
                            best_res = valid_res
                            best_step = step
                        elif  best_res > valid_res:
                            best_res = valid_res
                            best_step = step
                    if best_step + self.patience < step:
                        break
        if self.regression:
            return -best_res
        return best_res


    def get_rewards_default(self, architecture, global_step=''):
        tf.reset_default_graph()
        best_step = 0
        best_res = 0 if self.num_outputs > 1 else float('inf')
        metric_term = 'error' if self.num_outputs ==1 else 'accuracy'
        put_perc = '%' if self.num_outputs > 1 else ''
        with tf.Graph().as_default() as g:
            self.config['train_size'] = self.train_size
            model = BasicBuilder(self.config, architecture)
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                # if (self.config.get(a.summary)):
                if (True):
                    self.summary_writer = tf.summary.FileWriter(
                        'summary_{0}'.format(global_step), graph=tf.get_default_graph())
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
                    predictions, l, _ = sess.run([model.logits,
                                                  model.loss,
                                                  model.optimizer],
                                                  feed_dict=feed_dict)
                    if step % (self.eval_freq//10 if self.eval_freq//10 else 1) == 0:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        logs = 'Step %d (epoch %.2f), %.1f ms, ' % (step,
                            float(step) * self.batch_size / self.train_size,
                            1000 * elapsed_time / self.eval_freq)
                        logs += 'Minibatch loss: %.3f, ' % (l)
                        if put_perc:
                            logs += 'Minibatch %s: %.3f%%, ' %(metric_term,
                              model.test_metric(predictions, batch_labels))
                        else:
                            logs += 'Minibatch %s: %.3f%%, ' % (metric_term,
                                                           model.test_metric(predictions, batch_labels))
                        logger.debug(logs)

                    if step % self.eval_freq == 0:
                        valid_preds = self.eval_in_batches(
                            model, self.valid_X,  sess)
                        valid_res = model.test_metric(valid_preds, self.valid_y)
                        if put_perc:
                            logger.debug('Validation %s: %.3f%%' %(metric_term, valid_res))
                        else:
                            logger.debug('Validation %s: %.3f%%' %(metric_term, valid_res))

                        if not(self.regression) and best_res < valid_res:
                            best_res = valid_res
                            best_step = step
                        elif self.regression and best_res > valid_res:
                            best_res = valid_res
                            best_step = step
                        if best_step + self.patience < step:
                            logger.debug(f'--- PATIENCE BREAK ---- : best_step= {best_step} + patience= {self.patience} < step= {step}')
                            break
        if self.regression:
            return -best_res
        return best_res

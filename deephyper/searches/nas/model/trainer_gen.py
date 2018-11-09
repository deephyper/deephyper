import math
import time

import numpy as np
import tensorflow as tf

import deephyper.searches.nas.model.arch as a
from deephyper.searches import util
from deephyper.searches.nas.model.builder import BasicBuilder

tf.set_random_seed(1000003)
np.random.seed(1000003)

logger = util.conf_logger('deephyper.model.trainer.tf')

class GenTrainer:
    """
      GenTrainer class aims to train models
    """

    def __init__(self, config):
        logger.debug('[PARAM] Instantiate BasicTrainer')
        self.config = config
        self.config_hp = config[a.hyperparameters]

        self.input_shape = config[a.input_shape]
        self.output_shape = config[a.output_shape]

        self.num_outputs = self.output_shape[0] if config.get(a.num_outputs) is None else config.get(a.num_outputs) # supposing that the output is a vector for now

        self.regression = config[a.regression]
        self.data = config[a.data]
        self.batch_size = self.config_hp[a.batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]
        self.eval_freq = self.config_hp[a.eval_freq]

        self.gen_train = self.config[a.data]['gen_train']
        self.n_train = self.config[a.data]['n_train']
        self.gen_valid = self.config[a.data]['gen_valid']
        self.n_valid = self.config[a.data]['n_valid']

        self.train_size = self.n_train * self.num_epochs
        logger.debug('[PARAM] GenTrainer instantiated')

    def eval_one_batch(self, model, data, sess):
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
        logger.debug(f'eval_freq: {self.eval_freq}')
        with tf.Graph().as_default() as g:
            self.config['train_size'] = self.train_size
            model = BasicBuilder(self.config, architecture)
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                start_time = time.time()
                for step in range(self.num_epochs * self.n_train):
                    batch = next(self.gen_train)
                    batch_data = batch[0]
                    batch_labels = batch[1]
                    feed_dict = {model.train_data_node: batch_data,
                                 model.train_labels_node: batch_labels}
                    predictions, l, _ = sess.run([model.logits,
                                                  model.loss,
                                                  model.optimizer],
                                                  feed_dict=feed_dict)
                    if step % self.eval_freq == 0:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        logs = 'Step %d (epoch %.2f), %.1f ms, ' % (step,
                            float(step) / self.train_size,
                            1000 * elapsed_time / self.eval_freq)
                        logs += 'Minibatch loss: %.3f, ' % (l)
                        if put_perc:
                            logs += 'Minibatch %s: %.3f%%, ' %(metric_term,
                              model.test_metric(predictions, batch_labels))
                        else:
                            logs += 'Minibatch %s: %.3f%%, ' % (metric_term,
                                                           model.test_metric(predictions, batch_labels))
                        logger.debug(logs)

                    if (step % self.n_train == 0) and step != 0:
                        valid_res = 0
                        for n in range(self.n_valid):
                            batch = next(self.gen_valid)
                            valid_preds = self.eval_one_batch(
                                model, batch[0],  sess)
                            valid_res += model.test_metric(valid_preds, batch[1])
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
        if self.regression:
            return -best_res
        return best_res

import math
import time

import numpy as np
import tensorflow as tf

import deephyper.search.nas.model.arch as a
from deephyper.search import util
from deephyper.search.nas.model.builder import BasicBuilder

tf.set_random_seed(1000003)
np.random.seed(1000003)



logger = util.conf_logger('deephyper.model.trainer')

class BasicTrainer:
    """BasicTrainer class aims to train models
    """

    def __init__(self, config):
        logger.debug('[PARAM] Instantiate BasicTrainer')
        self.config = config
        self.config_hp = config[a.hyperparameters]

        self.input_shape = config[a.input_shape]
        self.output_shape = config[a.output_shape]

        self.num_outputs = self.output_shape[0]

        self.regression = config[a.regression]
        self.data = config[a.data]
        self.batch_size = self.config_hp[a.batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]

        # DATA loading
        self.train_X = None
        self.train_y = None
        self.valid_X = None
        self.valid_y = None
        self.train_size = None
        self.load_data()

        # DATA preprocessing
        self.preprocessing_func = None
        if self.config.get('preprocessing'):
            self.preprocessing_func = self.config['preprocessing']['func']
        self.preprocessor = None
        self.preprocess_data()

        # PATIENCE
        if a.patience in self.config_hp:
            self.patience = self.config_hp[a.patience]
        else:
            self.patience =  int(self.train_size/self.batch_size * self.num_epochs/5.)

        # Test on validation after each epoch
        self.eval_freq = self.train_size // self.batch_size
        self.model = None
        self.session = None
        logger.debug('[PARAM] BasicTrainer instantiated')

    def randomize_training_data(self):
        logger.debug('randomize_training_data')
        perm = np.random.permutation(np.shape(self.train_X)[0])
        self.train_X = self.train_X[perm]
        self.train_y = self.train_y[perm]

    def load_data(self):
        logger.debug('load_data')
        self.train_X = self.config[a.data][a.train_X]
        self.train_y = self.config[a.data][a.train_Y]
        self.valid_X = self.config[a.data][a.valid_X]
        self.valid_y = self.config[a.data][a.valid_Y]
        self.train_size = np.shape(self.config[a.data][a.train_X])[0]

    def preprocess_data(self):
        assert self.preprocessor is None, 'You can only preprocess the data one time.'

        if self.preprocessing_func:
            logger.debug('preprocess_data')
            data_train = np.concatenate((self.train_X, self.train_y), axis=1)
            data_valid = np.concatenate((self.valid_X, self.valid_y), axis=1)
            data = np.concatenate((data_train, data_valid), axis=0)
            self.preprocessor = self.preprocessing_func(data)

            t_X_shp = np.shape(self.train_X)

            preproc_data = self.preprocessor.fit_transform(data)

            self.train_X = preproc_data[:t_X_shp[0], :t_X_shp[1]]
            self.train_y = preproc_data[:t_X_shp[0], t_X_shp[1]:]
            self.valid_X = preproc_data[t_X_shp[0]:, :t_X_shp[1]]
            self.valid_y = preproc_data[t_X_shp[0]:, t_X_shp[1]:]
        else:
            logger.debug('no preprocessing function')

    def eval_in_batches(self, model, data, sess):
        size = data.shape[0]
        if size < self.batch_size:
            raise ValueError(
                'batch size for evals larger than dataset: %d' % size)
        predictions = np.ndarray(
            shape=([size] + self.output_shape), dtype=np.float32)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            if end > size:
                end = size
            predictions[begin:end, ...] = sess.run(model.eval_preds,
                feed_dict={
                    model.eval_data_node: data[begin:end, ...]})
        return predictions

    def predict_test(self):
        sess = self.session
        preds = sess.run(
            self.model.eval_preds,
            feed_dict={
                self.model.eval_data_node: self.valid_X
            })
        val_pred = np.concatenate((self.valid_X, preds), axis=1)
        val_orig = np.concatenate((self.valid_X, self.valid_y), axis=1)
        val_pred_trans = self.preprocessor.inverse_transform(val_pred)
        val_orig_trans = self.preprocessor.inverse_transform(val_orig)
        shp_X = np.shape(self.valid_X)
        valid_y = val_orig_trans[:, shp_X[1]:]
        pred_y  = val_pred_trans[:, shp_X[1]:]
        return valid_y, pred_y

    def predict_train(self):
        sess = self.session
        preds = sess.run(
            self.model.eval_preds,
            feed_dict={
                self.model.eval_data_node: self.train_X
            })
        val_pred = np.concatenate((self.train_X, preds), axis=1)
        val_orig = np.concatenate((self.train_X, self.train_y), axis=1)
        val_pred_trans = self.preprocessor.inverse_transform(val_pred)
        val_orig_trans = self.preprocessor.inverse_transform(val_orig)
        shp_X = np.shape(self.train_X)
        valid_y = val_orig_trans[:, shp_X[1]:]
        pred_y  = val_pred_trans[:, shp_X[1]:]
        return valid_y, pred_y

    def get_rewards(self, architecture):
        tf.reset_default_graph()
        best_step = 0
        best_res = 0 if self.num_outputs > 1 else float('inf')
        metric_term = 'error' if self.regression else 'accuracy'
        self.config['train_size'] = self.train_size

        self.model = BasicBuilder(self.config, architecture)
        self.session = tf.Session()

        summary_writer = tf.summary.FileWriter('logdir/', graph=tf.get_default_graph())


        init = tf.global_variables_initializer()
        self.session.run(init)
        start_time = time.time()

        self.randomize_training_data()

        num_steps = (self.num_epochs * self.train_size) // self.batch_size
        num_batch_per_epoch = self.train_size // self.batch_size
        loss_per_epoch = 0

        for step in range(num_steps):
            offset = (step * self.batch_size) % (self.train_size - self.batch_size)

            batch_data = self.train_X[offset:(
                                      offset + self.batch_size), ...]

            batch_labels = self.train_y[offset:(
                                        offset + self.batch_size)]

            feed_dict = {
                self.model.train_data_node: batch_data,
                self.model.train_labels_node: batch_labels
            }

            predictions, loss, g = self.session.run([
                                          self.model.logits,
                                          self.model.loss,
                                          self.model.optimizer
                                          ],
                                          feed_dict=feed_dict)
            loss_per_epoch += loss

            if step % (self.eval_freq//10 if self.eval_freq//10 else 1) == 0 \
                and step != 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()

                logs = 'Step %d (epoch %.2f), %.1f ms, ' % (step,
                    float(step) * self.batch_size / self.train_size,
                    1000 * elapsed_time / self.eval_freq)
                logs += 'Minibatch loss: %.3f, ' % (loss)

                if not self.regression:
                    logs += 'Minibatch %s: %.3f%%, ' %(metric_term,
                        self.model.test_metric(predictions, batch_labels))
                else:
                    logs += 'Minibatch %s: %.3f, ' % (metric_term,
                        self.model.test_metric(predictions, batch_labels))
                logger.debug(logs)

            if (step % self.eval_freq == 0): #and step != 0:

                pred_y = self.eval_in_batches(self.model, self.valid_X, self.session)

                # Reverse preprocessing if preprocessed was applied
                if self.preprocessing_func:
                    val_pred = np.concatenate((self.valid_X, pred_y), axis=1)
                    val_orig = np.concatenate((self.valid_X, self.valid_y), axis=1)
                    val_pred_trans = self.preprocessor.inverse_transform(val_pred)
                    val_orig_trans = self.preprocessor.inverse_transform(val_orig)
                    shp_X = np.shape(self.valid_X)
                    valid_y = val_orig_trans[:, shp_X[1]:]
                    pred_y  = val_pred_trans[:, shp_X[1]:]
                else:
                    valid_y = self.valid_y

                valid_res = self.model.test_metric(pred_y, valid_y)
                logs = f'avg_loss on last epoch = {loss_per_epoch / num_batch_per_epoch}'
                loss_per_epoch = 0

                if not self.regression:
                    logger.debug(logs+', Validation %s: %.3f%%' %(metric_term, valid_res))
                else:
                    logger.debug(logs+', Validation %s: %.3f' %(metric_term, valid_res))

                if not(self.regression) and best_res < valid_res:
                    best_res = valid_res
                    best_step = step
                elif self.regression and best_res > valid_res:
                    best_res = valid_res
                    best_step = step

                if best_step + self.patience < step:
                    logger.debug(f'--- PATIENCE BREAK ---- : best_step= {best_step} + patience= {self.patience} < step= {step}')
                    break
                self.randomize_training_data()
        if self.regression:
            return -best_res
        else:
            return best_res

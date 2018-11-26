import tensorflow as tf
import numpy as np
import math

from sklearn.metrics import mean_squared_error

import deephyper.search.nas.model.arch as a
import deephyper.search.nas.model.train_utils as U
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm

logger = util.conf_logger('deephyper.model.trainer')

class TrainerRegressorTrainValid:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.callbacks = []

        self.data = self.config[a.data]

        self.config_hp = self.config[a.hyperparameters]
        self.optimizer_name = self.config_hp[a.optimizer]
        self.loss_metric_name = self.config_hp[a.loss_metric]
        self.metrics_name = self.config_hp[a.metrics]
        self.batch_size = self.config_hp[a.batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]

        # DATA loading
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.train_size = None
        self.valid_size = None
        self.train_steps_per_epoch = None
        self.valid_steps_per_epoch = None
        self.load_data()

        # DATA preprocessing
        self.preprocessing_func = None
        if self.config.get('preprocessing'):
            self.preprocessing_func = self.config['preprocessing']['func']
        self.preprocessor = None
        self.preprocess_data()

        # Dataset
        self.dataset_train = None
        self.set_dataset_train()

        self.dataset_valid = None
        self.set_dataset_valid()

        self.model_compile()

        self.train_history = {
            f'{self.metrics_name[0]}_valid': list(),
        }
        # Test on validation after each epoch
        logger.debug('[PARAM] KerasTrainer instantiated')

    def load_data(self):
        logger.debug('load_data')
        self.train_X = self.config[a.data][a.train_X]
        self.train_Y = self.config[a.data][a.train_Y]
        self.valid_X = self.config[a.data][a.valid_X]
        self.valid_Y = self.config[a.data][a.valid_Y]
        self.train_size = np.shape(self.train_X)[0]
        self.valid_size = np.shape(self.valid_X)[0]
        self.train_steps_per_epoch = self.train_size // self.batch_size
        if self.train_steps_per_epoch * self.batch_size < self.train_size:
            self.train_steps_per_epoch += 1
        self.valid_steps_per_epoch = self.valid_size // self.batch_size
        if self.valid_steps_per_epoch * self.batch_size < self.valid_size:
            self.valid_steps_per_epoch += 1

    def preprocess_data(self):
        assert self.preprocessor is None, 'You can only preprocess the data one time.'

        if self.preprocessing_func:
            logger.debug('preprocess_data')
            data_train = np.concatenate((self.train_X, self.train_Y), axis=1)
            data_valid = np.concatenate((self.valid_X, self.valid_Y), axis=1)
            data = np.concatenate((data_train, data_valid), axis=0)
            self.preprocessor = self.preprocessing_func()

            t_X_shp = np.shape(self.train_X)

            preproc_data = self.preprocessor.fit_transform(data)

            self.train_X = preproc_data[:t_X_shp[0], :t_X_shp[1]]
            self.train_Y = preproc_data[:t_X_shp[0], t_X_shp[1]:]
            self.valid_X = preproc_data[t_X_shp[0]:, :t_X_shp[1]]
            self.valid_Y = preproc_data[t_X_shp[0]:, t_X_shp[1]:]
        else:
            logger.debug('no preprocessing function')

    def set_dataset_train(self):
        self.dataset_train = tf.data.Dataset.from_tensor_slices((self.train_X,
            self.train_Y))
        self.dataset_train = self.dataset_train.batch(self.batch_size).repeat()

    def set_dataset_valid(self):
        self.dataset_valid = tf.data.Dataset.from_tensor_slices((self.valid_X, self.valid_Y))
        self.dataset_valid = self.dataset_valid.batch(self.batch_size).repeat()

    def model_compile(self):
        optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        decay_rate = self.learning_rate / self.num_epochs
        self.optimizer = optimizer_fn(lr=self.learning_rate, decay=decay_rate)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_metric_name,
            metrics=self.metrics_name)

    def predict(self, dataset='valid'):
        assert dataset == 'valid' or dataset == 'train'
        if dataset == 'valid':
            y_pred = self.model.predict(self.dataset_valid, steps=self.valid_steps_per_epoch)
            data_X, data_Y = self.valid_X, self.valid_Y
        else:
            y_pred = self.model.predict(self.dataset_train,
            steps=self.train_steps_per_epoch)

        if self.preprocessing_func:
            val_pred = np.concatenate((data_X, y_pred), axis=1)
            val_orig = np.concatenate((data_X, data_Y), axis=1)
            val_pred_trans = self.preprocessor.inverse_transform(val_pred)
            val_orig_trans = self.preprocessor.inverse_transform(val_orig)
            shp_X = np.shape(data_X)
            y_orig = val_orig_trans[:, shp_X[1]:]
            y_pred  = val_pred_trans[:, shp_X[1]:]
        else:
            y_orig = self.valid_Y

        return y_orig, y_pred

    def train(self, num_epochs=None):
        num_epochs = self.num_epochs if num_epochs is None else num_epochs

        min_mse = math.inf
        for i in range(num_epochs):
            self.model.fit(
                self.dataset_train,
                epochs=1,
                steps_per_epoch=self.train_steps_per_epoch,
                callbacks=self.callbacks
            )

            y_orig, y_pred = self.predict()

            unnormalize_mse = mean_squared_error(y_orig, y_pred)

            self.train_history[f'{self.metrics_name[0]}_valid'] = unnormalize_mse

            min_mse = min(min_mse, unnormalize_mse)
            logger.info(jm(epoch=i, validation_mse=float(unnormalize_mse)))

        logger.info(jm(type='result', mse=float(min_mse)))
        return min_mse

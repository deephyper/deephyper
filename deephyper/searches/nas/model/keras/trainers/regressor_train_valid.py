import tensorflow as tf
import numpy as np
import math

from sklearn.metrics import mean_squared_error

import deephyper.searches.nas.model.arch as a
import deephyper.searches.nas.model.train_utils as U
from deephyper.searches import util
from deephyper.searches.nas.utils._logging import JsonMessage as jm

logger = util.conf_logger('deephyper.model.trainer')

class KerasTrainerRegressorTrainValid:
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

        # PATIENCE
        if a.patience in self.config_hp:
            self.patience = self.config_hp[a.patience]
        else:
            self.patience =  int(self.train_size/self.batch_size * self.num_epochs/5.)

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

    def preprocess_data(self):
        assert self.preprocessor is None, 'You can only preprocess the data one time.'

        if self.preprocessing_func:
            logger.debug('preprocess_data')
            data_train = np.concatenate((self.train_X, self.train_Y), axis=1)
            data_valid = np.concatenate((self.valid_X, self.valid_Y), axis=1)
            data = np.concatenate((data_train, data_valid), axis=0)
            self.preprocessor = self.preprocessing_func(data)

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
        self.dataset_train = self.dataset_train.batch(self.batch_size)
        self.dataset_train = self.dataset_train.repeat()

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

    def train(self):
        train_steps_per_epoch = self.train_size // self.batch_size
        valid_steps_per_epoch = self.valid_size // self.batch_size

        min_mse = math.inf
        for i in range(self.num_epochs):
            self.model.fit(
                self.dataset_train,
                epochs=1,
                steps_per_epoch=train_steps_per_epoch,
                callbacks=self.callbacks
            )

            y_pred = self.model.predict(self.dataset_valid, steps=valid_steps_per_epoch)

            if self.preprocessing_func:
                val_pred = np.concatenate((self.valid_X, y_pred), axis=1)
                val_orig = np.concatenate((self.valid_X, self.valid_Y), axis=1)
                val_pred_trans = self.preprocessor.inverse_transform(val_pred)
                val_orig_trans = self.preprocessor.inverse_transform(val_orig)
                shp_X = np.shape(self.valid_X)
                y_orig = val_orig_trans[:, shp_X[1]:]
                y_pred  = val_pred_trans[:, shp_X[1]:]
            else:
                y_orig = self.valid_Y
            unnormalize_mse = mean_squared_error(y_orig, y_pred)
            min_mse = min(min_mse, unnormalize_mse)
            logger.info(jm(epoch=i, validation_mse=float(unnormalize_mse)))
        logger.info(jm(type='result', mse=float(min_mse)))
        return min_mse

import tensorflow as tf
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import deephyper.search.nas.model.arch as a
import deephyper.search.nas.model.train_utils as U
from deephyper.search import util

logger = util.conf_logger('deephyper.model.trainer')

class KerasTrainerRegressorKfold:
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
        self.data_X = None
        self.data_Y = None
        self.train_size = None
        self.load_data()

        # DATA preprocessing
        self.preprocessing_func = None
        if self.config.get('preprocessing'):
            self.preprocessing_func = self.config['preprocessing']['func']
        self.preprocessor = None
        # self.preprocess_data()

        # Dataset
        self.dataset_train = None
        self.set_dataset_train()

        self.dataset_valid = None
        self.set_dataset_valid()

        self.set_early_stopping()

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
        train_X = self.config[a.data][a.train_X]
        train_Y = self.config[a.data][a.train_Y]
        valid_X = self.config[a.data][a.valid_X]
        valid_Y = self.config[a.data][a.valid_Y]
        self.data_X = np.concatenate((train_X, valid_X), axis=0)
        self.data_Y = np.concatenate((train_Y, valid_Y), axis=0)
        self.train_size = np.shape(self.data_X)[0]

    def preprocess_data(self):
        assert self.preprocessor is None, 'You can only preprocess the data one time.'

        if self.preprocessing_func:
            logger.debug('preprocess_data')
            data_train = np.concatenate((self.train_X, self.train_y), axis=1)
            data_valid = np.concatenate((self.valid_X, self.valid_y), axis=1)
            data = np.concatenate((data_train, data_valid), axis=0)
            self.preprocessor = self.preprocessing_func()

            t_X_shp = np.shape(self.train_X)

            preproc_data = self.preprocessor.fit_transform(data)

            self.train_X = preproc_data[:t_X_shp[0], :t_X_shp[1]]
            self.train_y = preproc_data[:t_X_shp[0], t_X_shp[1]:]
            self.valid_X = preproc_data[t_X_shp[0]:, :t_X_shp[1]]
            self.valid_y = preproc_data[t_X_shp[0]:, t_X_shp[1]:]
        else:
            logger.debug('no preprocessing function')

    def set_dataset_train(self):
        self.dataset_train = tf.data.Dataset.from_tensor_slices((self.train_X,
            self.train_y))
        self.dataset_train = self.dataset_train.batch(self.batch_size)
        self.dataset_train = self.dataset_train.repeat()

    def set_dataset_valid(self):
        self.dataset_valid = tf.data.Dataset.from_tensor_slices((self.valid_X, self.valid_y))
        self.dataset_valid = self.dataset_valid.batch(self.batch_size).repeat()

    def set_early_stopping(self):
        if not self.regression:
            earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
            self.callbacks.append(earlystop)

    def model_compile(self):
        optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        self.optimizer = optimizer_fn(lr=self.learning_rate, decay=0.0)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_metric_name,
            metrics=self.metrics_name)

    def train(self):
        # steps_per_epoch = self.train_size // self.batch_size
        # self.model_info = self.model.fit(
        #     self.dataset_train,
        #     epochs=1,
        #     steps_per_epoch=steps_per_epoch)

        # self.model.predict(self.data)

        estimators = []
        estimators.append(('standardize', StandardScaler()))

        estimators.append(('model', tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=lambda: self.model, epochs=10, batch_size=self.batch_size, verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=2018)
        results = cross_val_score(pipeline, self.train_X, self.train_y, cv=kfold)

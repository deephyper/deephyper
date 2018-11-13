import tensorflow as tf
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import deephyper.searches.nas.model.arch as a
import deephyper.searches.nas.model.train_utils as U
from deephyper.searches import util

logger = util.conf_logger('deephyper.model.trainer')

class KerasTrainerRegressorKfold:
    def __init__(self, config, model, seed=2018):
        self.seed = seed
        np.random.seed(seed)

        self.config = config
        self.model = model

        self.data = self.config[a.data]
        self.config_hp = self.config[a.hyperparameters]
        self.optimizer_name = self.config_hp[a.optimizer]
        self.loss_metric_name = self.config_hp[a.loss_metric]
        self.metrics_name = self.config_hp[a.metrics]
        self.batch_size = self.config_hp[a.batch_size]
        self.learning_rate = self.config_hp[a.learning_rate]
        self.num_epochs = self.config_hp[a.num_epochs]
        self.n_split_kfold = self.config_hp['n_split_kfold']

        # DATA loading
        self.data_X = None
        self.data_Y = None
        self.train_size = None
        self.load_data()

        self.model_compile()

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

    def model_compile(self):
        optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        self.optimizer = optimizer_fn(lr=self.learning_rate, decay=0.0)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_metric_name,
            metrics=self.metrics_name)

    def train(self):
        estimators = []
        estimators.append(('standardize', StandardScaler()))

        estimators.append(('model',
            tf.keras.wrappers.scikit_learn.KerasRegressor(
                build_fn=lambda: self.model,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=self.n_split_kfold, random_state=self.seed)
        results = cross_val_score(pipeline, self.data_X, self.data_Y, cv=kfold)
        print(results)

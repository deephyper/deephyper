import tensorflow as tf
import numpy as np

import deephyper.search.nas.model.arch as a
import deephyper.search.nas.model.train_utils as U
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm

logger = util.conf_logger('deephyper.model.trainer')

class TrainerClassifierTrainValid:
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

        # Dataset
        self.dataset_train = None
        self.set_dataset_train()

        self.dataset_valid = None
        self.set_dataset_valid()

        self.model_compile()

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

        print('compile')
        print('optimizer: ', self.optimizer)
        print('loss: ', self.loss_metric_name)
        print('metrics: ', self.metrics_name)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_metric_name,
            metrics=self.metrics_name)

    def predict(self, dataset='valid'):
        assert dataset == 'valid' or dataset == 'train'
        if dataset == 'valid':
            y_pred = self.model.predict(self.dataset_valid, steps=self.valid_steps_per_epoch)
        else:
            y_pred = self.model.predict(self.dataset_train,
            steps=self.train_steps_per_epoch)

        y_orig = self.valid_Y

        return y_orig, y_pred

    def train(self):
        max_acc = 0
        for i in range(self.num_epochs):
            self.model.fit(
                self.dataset_train,
                epochs=1,
                steps_per_epoch=self.train_steps_per_epoch,
                callbacks=self.callbacks
            )

            valid_info = self.model.evaluate(self.dataset_valid, steps=self.valid_steps_per_epoch)

            valid_loss, valid_acc = valid_info[0], valid_info[1]*100

            max_acc = max(max_acc, valid_acc)
            logger.info(jm(epoch=i, validation_loss=valid_loss, validation_acc=float(valid_acc)))
        logger.info(jm(type='result', acc=float(max_acc)))
        return max_acc

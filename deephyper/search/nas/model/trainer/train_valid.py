import math
import time
import traceback
from inspect import signature

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from .....core.logs.logging import JsonMessage as jm
from .... import util
from .. import arch as a
from .. import train_utils as U

logger = util.conf_logger('deephyper.model.trainer')




class TrainerTrainValid:
    def __init__(self, config, model):
        self.cname = self.__class__.__name__

        self.config = config
        self.sess = keras.backend.get_session()
        self.model = model
        self.callbacks = [
            keras.callbacks.CSVLogger('training.csv', append=True)
        ]

        self.data = self.config[a.data]

        self.config_hp = self.config[a.hyperparameters]
        self.optimizer_name = self.config_hp.get(a.optimizer, 'adam')
        self.optimizer_eps = self.config_hp.get('epsilon', None)
        self.batch_size = self.config_hp.get(a.batch_size, 32)
        self.learning_rate = self.config_hp.get(a.learning_rate, 1e-3)
        self.num_epochs = self.config_hp[a.num_epochs]
        self.verbose = self.config_hp.get('verbose', 1)

        self.loss_metric_name = self.config[a.loss_metric]
        self.metrics_name = [U.selectMetric(m) for m in self.config[a.metrics]]

        # DATA loading
        self.data_config_type = None
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

        self.train_history = None
        self.init_history()

        # Test on validation after each epoch
        logger.debug('[PARAM] KerasTrainer instantiated')

    def init_history(self):
        self.train_history = dict()
        self.train_history['n_parameters'] = self.model.count_params()

    def load_data(self):
        logger.debug('load_data')

        self.data_config_type = U.check_data_config(self.data)
        logger.debug(f'data config type: {self.data_config_type}')
        if self.data_config_type == 'gen':
            self.load_data_generator()
        elif self.data_config_type == 'ndarray':
            self.load_data_ndarray()
        else:
            raise RuntimeError(
                f"Data config is not supported by this Trainer: '{self.data_config_type}'!")

        # prepare number of steps for training and validation
        self.train_steps_per_epoch = self.train_size // self.batch_size
        if self.train_steps_per_epoch * self.batch_size < self.train_size:
            self.train_steps_per_epoch += 1
        self.valid_steps_per_epoch = self.valid_size // self.batch_size
        if self.valid_steps_per_epoch * self.batch_size < self.valid_size:
            self.valid_steps_per_epoch += 1

    def load_data_generator(self):
        self.train_gen = self.data["train_gen"]
        self.valid_gen = self.data["valid_gen"]
        self.data_types = self.data["types"]
        self.data_shapes = self.data["shapes"]

        self.train_size = self.data["train_size"]
        self.valid_size = self.data["valid_size"]

    def load_data_ndarray(self):
        # check data type
        if not type(self.config[a.data][a.train_Y]) is np.ndarray:
            raise RuntimeError(
                f"train_Y data should be of type np.ndarray when type true type is: {type(self.config[a.data][a.train_Y])}")
        self.train_Y = self.config[a.data][a.train_Y]

        if not type(self.config[a.data][a.valid_Y]) is np.ndarray:
            raise RuntimeError(
                f"valid_Y data should be of type np.ndarray when type true type is: {type(self.config[a.data][a.valid_Y])}")
        self.valid_Y = self.config[a.data][a.valid_Y]

        if type(self.config[a.data][a.train_X]) is np.ndarray and \
                type(self.config[a.data][a.valid_X]) is np.ndarray:
            self.train_X = [self.config[a.data][a.train_X]]
            self.valid_X = [self.config[a.data][a.valid_X]]
        elif type(self.config[a.data][a.train_X]) is list and \
                type(self.config[a.data][a.valid_X]) is list:
            def f(x): return type(x) is np.ndarray
            if not all(map(f, self.config[a.data][a.train_X])) or \
                    not all(map(f, self.config[a.data][a.valid_X])):
                raise RuntimeError(
                    f"all inputs data should be of type np.ndarray !")
            self.train_X = self.config[a.data][a.train_X]
            self.valid_X = self.config[a.data][a.valid_X]
        else:
            raise RuntimeError(
                f"Data are of an unsupported type and should be of same type: type(self.config['data']['train_X'])=={type(self.config[a.data])} and type(self.config['data']['valid_X'])=={type(self.config[a.data][a.valid_X])} !")

        logger.debug(f'{self.cname}: {len(self.train_X)} inputs')

        # check data length
        self.train_size = np.shape(self.train_X[0])[0]
        if not all(map(lambda x: np.shape(x)[0] == self.train_size, self.train_X)):
            raise RuntimeError(
                f'All training inputs data should have same length!')

        self.valid_size = np.shape(self.valid_X[0])[0]
        if not all(map(lambda x: np.shape(x)[0] == self.valid_size, self.valid_X)):
            raise RuntimeError(
                f'All validation inputs data should have same length!')

    def preprocess_data(self):
        if self.data_config_type == "gen":
            return

        if not self.preprocessor is None:
            raise RuntimeError('You can only preprocess data one time.')

        if self.preprocessing_func:
            logger.debug(
                f'preprocess_data with: {str(self.preprocessing_func)}')

            data_train = np.concatenate((*self.train_X, self.train_Y), axis=1)
            data_valid = np.concatenate((*self.valid_X, self.valid_Y), axis=1)
            data = np.concatenate((data_train, data_valid), axis=0)
            self.preprocessor = self.preprocessing_func()

            dt_shp = np.shape(data_train)
            tX_shp = [np.shape(x) for x in self.train_X]

            preproc_data = self.preprocessor.fit_transform(data)

            acc, self.train_X = 0, list()
            for shp in tX_shp:
                self.train_X.append(preproc_data[:dt_shp[0], acc:acc+shp[1]])
                acc += shp[1]
            self.train_Y = preproc_data[:dt_shp[0], acc:]

            acc, self.valid_X = 0, list()
            for shp in tX_shp:
                self.valid_X.append(preproc_data[dt_shp[0]:, acc:acc+shp[1]])
                acc += shp[1]
            self.valid_Y = preproc_data[dt_shp[0]:, acc:]
        else:
            logger.info('no preprocessing function')

    def set_dataset_train(self):
        if self.data_config_type == "ndarray":
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                {f'input_{i}': tX for i, tX in enumerate(self.train_X)},
                self.train_Y))
        else:  # self.data_config_type == "gen"
            self.dataset_train = tf.data.Dataset.from_generator(self.train_gen,
                                                                output_types=self.data_types,
                                                                output_shapes=({
                                                                    f'input_{i}': tf.TensorShape([*self.data_shapes[0][f'input_{i}']]) for i in range(len(self.data_shapes[0]))
                                                                }, tf.TensorShape([*self.data_shapes[1]])))
        self.dataset_train = self.dataset_train.batch(self.batch_size).repeat()

    def set_dataset_valid(self):
        if self.data_config_type == "ndarray":
            self.dataset_valid = tf.data.Dataset.from_tensor_slices((
                {f'input_{i}': vX for i, vX in enumerate(self.valid_X)},
                self.valid_Y))
        else:
            self.dataset_valid = tf.data.Dataset.from_generator(self.valid_gen,
                                                                output_types=self.data_types,
                                                                output_shapes=({
                                                                    f'input_{i}': tf.TensorShape([*self.data_shapes[0][f'input_{i}']]) for i in range(len(self.data_shapes[0]))
                                                                }, tf.TensorShape([*self.data_shapes[1]])))
        self.dataset_valid = self.dataset_valid.batch(self.batch_size).repeat()

    def model_compile(self):
        optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        decay_rate = self.learning_rate / \
            self.num_epochs if self.num_epochs > 0 else 1

        opti_parameters = signature(optimizer_fn).parameters
        params = {}
        if 'lr' in opti_parameters:
            params['lr'] = self.learning_rate
        if 'epsilon' in opti_parameters:
            params['epsilon'] = self.optimizer_eps
        if 'decay' in opti_parameters:
            decay_rate = self.learning_rate / \
                self.num_epochs if self.num_epochs > 0 else 1
            params['decay'] = decay_rate
        self.optimizer = optimizer_fn(**params)


        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_metric_name,
            metrics=self.metrics_name)

    def predict(self, dataset: str='valid', keep_normalize: bool=False) -> tuple:
        """[summary]

        Args:
            dataset (str, optional): 'valid' or 'train'. Defaults to 'valid'.
            keep_normalize (bool, optional): if False then the preprocessing will be reversed after prediction. if True nothing will be reversed. Defaults to False.

        Raises:
            RuntimeError: [description]

        Returns:
            tuple: (y_true, y_pred)
        """
        if not(dataset == 'valid' or dataset == 'train'):
            raise RuntimeError(
                "dataset parameter should be equal to: 'valid' or 'train'")

        if dataset == 'valid':
            y_pred = self.model.predict(
                self.dataset_valid, steps=self.valid_steps_per_epoch)
        else:  # dataset == 'train'
            y_pred = self.model.predict(self.dataset_train,
                                        steps=self.train_steps_per_epoch)

        if self.preprocessing_func and not keep_normalize and \
                not self.data_config_type == "gen":
            if dataset == 'valid':
                data_X, data_Y = self.valid_X, self.valid_Y
            else:  # dataset == 'train'
                data_X, data_Y = self.train_X, self.train_Y
            val_pred = np.concatenate((*data_X, y_pred), axis=1)
            val_orig = np.concatenate((*data_X, data_Y), axis=1)
            val_pred_trans = self.preprocessor.inverse_transform(val_pred)
            val_orig_trans = self.preprocessor.inverse_transform(val_orig)
            y_orig = val_orig_trans[:, -np.shape(data_Y)[1]:]
            y_pred = val_pred_trans[:, -np.shape(data_Y)[1]:]
        else:
            if self.data_config_type == "ndarray":
                y_orig = self.valid_Y if dataset == "valid" else self.train_Y
            else:
                gen = self.valid_gen() if dataset == "valid" else self.train_gen()
                y_orig = np.array([e[-1] for e in gen])

        return y_orig, y_pred

    def evaluate(self, dataset='train'):
        """Evaluate the performance of your model for the same configuration.

        Args:
            dataset (str, optional): must be "train" or "valid". If "train" then metrics will be evaluated on the training dataset. If "valid" then metrics will be evaluated on the "validation" dataset. Defaults to 'train'.

        Returns:
            list: a list of scalar values corresponding do config loss & metrics.
        """
        if dataset == 'train':
            return self.model.evaluate(self.dataset_train,
                        steps=self.train_steps_per_epoch)
        else:
            return self.model.evaluate(self.dataset_valid,
                        steps=self.valid_steps_per_epoch)

    def train(self, num_epochs: int=None, with_pred: bool=False, last_only: bool=False):
        """Train the model.

        Args:
            num_epochs (int, optional): override the num_epochs passed to init the Trainer. Defaults to None, will use the num_epochs passed to init the Trainer.
            with_pred (bool, optional): will compute a prediction after the training and will add ('y_true', 'y_pred') to the output history. Defaults to False, will skip it (use it to save compute time).
            last_only (bool, optional): will compute metrics after the last epoch only. Defaults to False, will compute metrics after each training epoch (use it to save compute time).

        Raises:
            RuntimeError: raised when the ``num_epochs < 0``.

        Returns:
            dict: a dictionnary corresponding to the training.
        """
        num_epochs = self.num_epochs if num_epochs is None else num_epochs

        self.init_history()

        if num_epochs > 0:

            time_start_training = time.time()  # TIMING

            if not last_only:
                logger.info('Trainer is computing metrics on validation after each training epoch.')
                history = self.model.fit(
                    self.dataset_train,
                    verbose=self.verbose,
                    epochs=num_epochs,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch
                )
            else:
                logger.info('Trainer is computing metrics on validation after the last training epoch.')
                if num_epochs > 1:
                    self.model.fit(
                        self.dataset_train,
                        verbose=self.verbose,
                        epochs=num_epochs-1,
                        steps_per_epoch=self.train_steps_per_epoch,
                        callbacks=self.callbacks,
                    )
                history = self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    verbose=self.verbose,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch
                )

            time_end_training = time.time()  # TIMING
            self.train_history['training_time'] = time_end_training - \
                time_start_training

            self.train_history.update(history.history)

        elif num_epochs < 0:

            raise RuntimeError(
                f'Trainer: number of epochs should be >= 0: {num_epochs}')

        if  with_pred:
            y_true, y_pred= self.predict()

            self.train_history['y_true'] = y_true
            self.train_history['y_pred'] = y_pred

        return self.train_history

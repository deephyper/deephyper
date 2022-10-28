import inspect
import logging
import time
from inspect import signature

import deephyper.nas.trainer._arch as a
import deephyper.nas.trainer._utils as U
import numpy as np
import tensorflow as tf
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.nas.losses import selectLoss
from deephyper.nas.metrics import selectMetric

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, model):
        self.cname = self.__class__.__name__

        self.config = config

        self.model = model
        self.callbacks = []

        self.data = self.config[a.data]

        self.config_hp = self.config[a.hyperparameters]
        self.optimizer_name = self.config_hp.get(a.optimizer, "adam")
        self.optimizer_eps = self.config_hp.get("epsilon", None)
        self.batch_size = self.config_hp.get(a.batch_size, 32)
        self.learning_rate = self.config_hp.get(a.learning_rate, 1e-3)
        self.num_epochs = self.config_hp.get(a.num_epochs, 1)
        self.shuffle_data = self.config_hp.get(a.shuffle_data, True)
        self.cache_data = self.config_hp.get(a.cache_data, True)
        self.batch = self.config_hp.get("batch", True)
        self.momentum = self.config_hp.get("momentum", 0.0)
        self.nesterov = self.config_hp.get("nesterov", False)
        self.label_smoothing = self.config_hp.get("label_smoothing", 0.0)
        self.verbose = self.config_hp.get("verbose", 1)
        # self.balanced = self.config_hp.get("balanced", False)

        self.setup_losses_and_metrics()

        # DATA loading
        self.data_config_type = None
        self.train_size = None
        self.valid_size = None
        self.train_steps_per_epoch = None
        self.valid_steps_per_epoch = None
        self.load_data()

        # DATA preprocessing
        self.preprocessing_func = None
        if self.config.get("preprocessing"):
            self.preprocessing_func = self.config["preprocessing"]["func"]
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
        if self.verbose == 1:
            logger.info("KerasTrainer instantiated")
            model.summary(print_fn=logger.info)

    def init_history(self):
        self.train_history = dict()
        self.train_history["n_parameters"] = self.model.count_params()

    def _select_loss(self, loss):
        if type(loss) is dict:
            loss = {k: selectLoss(v) for k, v in loss.items()}
        else:
            loss = selectLoss(loss)

        if inspect.isclass(loss):

            loss_parameters = signature(loss).parameters
            params = {}

            if "label_smoothing" in loss_parameters:
                params["label_smoothing"] = self.label_smoothing

            loss = loss(**params)

        return loss

    def setup_losses_and_metrics(self):

        self.loss_metrics = self._select_loss(self.config[a.loss_metric])
        self.loss_weights = self.config.get("loss_weights")
        self.class_weights = self.config.get("class_weights")

        if self.loss_weights is None and type(self.loss_metrics) is dict:
            self.loss_weights = [1.0 for _ in range(len(self.loss_metrics))]

        if type(self.config[a.metrics]) is list:
            self.metrics_name = [selectMetric(m) for m in self.config[a.metrics]]
        else:

            def selectM(metric):
                if type(metric) is list:
                    return [selectMetric(m_i) for m_i in metric]
                else:
                    return selectMetric(metric)

            self.metrics_name = {
                n: selectM(m) for n, m in self.config[a.metrics].items()
            }

    def load_data(self):
        logger.debug("load_data")

        self.data_config_type = U.check_data_config(self.data)
        logger.debug(f"data config type: {self.data_config_type}")
        if self.data_config_type == "gen":
            self.load_data_generator()
        elif self.data_config_type == "ndarray":
            self.load_data_ndarray()
        else:
            raise DeephyperRuntimeError(
                f"Data config is not supported by this Trainer: '{self.data_config_type}'!"
            )

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
        def f(x):
            return type(x) is np.ndarray

        # check data type

        # Output data
        if (
            type(self.config[a.data][a.train_Y]) is np.ndarray
            and type(self.config[a.data][a.valid_Y]) is np.ndarray
        ):
            self.train_Y = self.config[a.data][a.train_Y]
            self.valid_Y = self.config[a.data][a.valid_Y]
        elif (
            type(self.config[a.data][a.train_Y]) is list
            and type(self.config[a.data][a.valid_Y]) is list
        ):

            if not all(map(f, self.config[a.data][a.train_Y])) or not all(
                map(f, self.config[a.data][a.valid_Y])
            ):
                raise DeephyperRuntimeError(
                    "all outputs data should be of type np.ndarray !"
                )

            if (
                len(self.config[a.data][a.train_Y]) > 1
                and len(self.config[a.data][a.valid_Y]) > 1
            ):
                self.train_Y = self.config[a.data][a.train_Y]
                self.valid_Y = self.config[a.data][a.valid_Y]
            else:
                self.train_Y = self.config[a.data][a.train_Y][0]
                self.valid_Y = self.config[a.data][a.valid_Y][0]
        else:
            raise DeephyperRuntimeError(
                f"Data are of an unsupported type and should be of same type: type(self.config['data']['train_Y'])=={type(self.config[a.data][a.train_Y])} and type(self.config['data']['valid_Y'])=={type(self.config[a.valid_Y][a.valid_X])} !"
            )

        # Input data
        if (
            type(self.config[a.data][a.train_X]) is np.ndarray
            and type(self.config[a.data][a.valid_X]) is np.ndarray
        ):
            self.train_X = [self.config[a.data][a.train_X]]
            self.valid_X = [self.config[a.data][a.valid_X]]
        elif (
            type(self.config[a.data][a.train_X]) is list
            and type(self.config[a.data][a.valid_X]) is list
        ):

            if not all(map(f, self.config[a.data][a.train_X])) or not all(
                map(f, self.config[a.data][a.valid_X])
            ):
                raise DeephyperRuntimeError(
                    "all inputs data should be of type np.ndarray !"
                )
            if (
                len(self.config[a.data][a.train_X]) > 1
                and len(self.config[a.data][a.valid_X]) > 1
            ):
                self.train_X = self.config[a.data][a.train_X]
                self.valid_X = self.config[a.data][a.valid_X]
            else:
                self.train_X = self.config[a.data][a.train_X][0]
                self.valid_X = self.config[a.data][a.valid_X][0]
        else:
            raise DeephyperRuntimeError(
                f"Data are of an unsupported type and should be of same type: type(self.config['data']['train_X'])=={type(self.config[a.data][a.train_X])} and type(self.config['data']['valid_X'])=={type(self.config[a.data][a.valid_X])} !"
            )

        logger.debug(f"{self.cname}: {len(self.train_X)} inputs")

        # check data length
        self.train_size = np.shape(self.train_X[0])[0]
        if not all(map(lambda x: np.shape(x)[0] == self.train_size, self.train_X)):
            raise DeephyperRuntimeError(
                "All training inputs data should have same length!"
            )

        self.valid_size = np.shape(self.valid_X[0])[0]
        if not all(map(lambda x: np.shape(x)[0] == self.valid_size, self.valid_X)):
            raise DeephyperRuntimeError(
                "All validation inputs data should have same length!"
            )

    def preprocess_data(self):
        logger.debug("Starting preprocess of data")

        if self.data_config_type == "gen":
            logger.warn("Cannot preprocess data with generator!")
            return

        if self.preprocessor is not None:
            raise DeephyperRuntimeError("You can only preprocess data one time.")

        if self.preprocessing_func:
            logger.debug(f"preprocess_data with: {str(self.preprocessing_func)}")
            if all(
                [
                    len(np.shape(tX)) == len(np.shape(self.train_Y))
                    for tX in self.train_X
                ]
            ):
                data_train = np.concatenate((*self.train_X, self.train_Y), axis=-1)
                data_valid = np.concatenate((*self.valid_X, self.valid_Y), axis=-1)
                self.preprocessor = self.preprocessing_func()

                tX_shp = [np.shape(x) for x in self.train_X]

                preproc_data_train = self.preprocessor.fit_transform(data_train)
                preproc_data_valid = self.preprocessor.transform(data_valid)

                acc, self.train_X = 0, list()
                for shp in tX_shp:
                    self.train_X.append(preproc_data_train[..., acc : acc + shp[1]])
                    acc += shp[1]
                self.train_Y = preproc_data_train[..., acc:]

                acc, self.valid_X = 0, list()
                for shp in tX_shp:
                    self.valid_X.append(preproc_data_valid[..., acc : acc + shp[1]])
                    acc += shp[1]
                self.valid_Y = preproc_data_valid[..., acc:]
            else:
                logger.warn(
                    f"Skipped preprocess because shape {np.shape(self.train_Y)} is not handled!"
                )
        else:
            logger.info("Skipped preprocess of data because no function is defined!")

    def set_dataset_train(self):
        if self.data_config_type == "ndarray":
            if type(self.train_Y) is list:
                output_mapping = {
                    f"output_{i}": tY for i, tY in enumerate(self.train_Y)
                }
            else:
                output_mapping = self.train_Y
            self.dataset_train = tf.data.Dataset.from_tensor_slices(
                (
                    {f"input_{i}": tX for i, tX in enumerate(self.train_X)},
                    output_mapping,
                )
            )
        else:  # self.data_config_type == "gen"
            self.dataset_train = tf.data.Dataset.from_generator(
                self.train_gen,
                output_signature=self._get_output_signatures(),
            )

        if self.cache_data:
            self.dataset_train = self.dataset_train.cache()
        if self.shuffle_data:
            self.dataset_train = self.dataset_train.shuffle(
                self.train_size, reshuffle_each_iteration=True
            )
        if self.batch:
            self.dataset_train = self.dataset_train.batch(self.batch_size)

        self.dataset_train = self.dataset_train.prefetch(tf.data.AUTOTUNE).repeat(
            self.num_epochs
        )

    def set_dataset_valid(self):
        if self.data_config_type == "ndarray":
            if type(self.valid_Y) is list:
                output_mapping = {
                    f"output_{i}": vY for i, vY in enumerate(self.valid_Y)
                }
            else:
                output_mapping = self.valid_Y
            self.dataset_valid = tf.data.Dataset.from_tensor_slices(
                (
                    {f"input_{i}": vX for i, vX in enumerate(self.valid_X)},
                    output_mapping,
                )
            )
        else:
            self.dataset_valid = tf.data.Dataset.from_generator(
                self.valid_gen,
                output_signature=self._get_output_signatures(valid=True),
            )

        self.dataset_valid = self.dataset_valid.cache()
        self.dataset_valid = self.dataset_valid.batch(self.batch_size)
        self.dataset_valid = self.dataset_valid.prefetch(tf.data.AUTOTUNE).repeat(
            self.num_epochs
        )

    def _get_output_signatures(self, valid=False):
        if self.batch or valid:
            return (
                {
                    f"input_{i}": tf.TensorSpec(
                        shape=(*self.data_shapes[0][f"input_{i}"],),
                        dtype=self.data_types[0][f"input_{i}"],
                    )
                    for i in range(len(self.data_shapes[0]))
                },
                tf.TensorSpec(
                    shape=(*self.data_shapes[1],),
                    dtype=self.data_types[1],
                ),
            )
        else:
            return (
                {
                    f"input_{i}": tf.TensorSpec(
                        shape=(
                            None,
                            *self.data_shapes[0][f"input_{i}"],
                        ),
                        dtype=self.data_types[0][f"input_{i}"],
                    )
                    for i in range(len(self.data_shapes[0]))
                },
                tf.TensorSpec(
                    shape=(None, *self.data_shapes[1]),
                    dtype=self.data_types[1],
                ),
            )

    def _setup_optimizer(self):
        optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        opti_parameters = signature(optimizer_fn).parameters
        params = {}

        if "lr" in opti_parameters:
            params["lr"] = self.learning_rate
        elif "learning_rate" in opti_parameters:
            params["learning_rate"] = self.learning_rate
        else:
            raise DeephyperRuntimeError(
                f"The learning_rate parameter is not found amoung optimiser arguments: {opti_parameters}"
            )

        if "epsilon" in opti_parameters:
            params["epsilon"] = self.optimizer_eps

        if "momentum" in opti_parameters:
            params["momentum"] = self.momentum

        if "nesterov" in opti_parameters:
            params["nesterov"] = self.nesterov

        self.optimizer = optimizer_fn(**params)

    def model_compile(self):

        self._setup_optimizer()

        if type(self.loss_metrics) is dict:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_metrics,
                loss_weights=self.loss_weights,
                metrics=self.metrics_name,
            )
        else:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_metrics,
                metrics=self.metrics_name,
            )

    def predict(self, dataset: str = "valid", keep_normalize: bool = False) -> tuple:
        """[summary]

        Args:
            dataset (str, optional): 'valid' or 'train'. Defaults to 'valid'.
            keep_normalize (bool, optional): if False then the preprocessing will be reversed after prediction. if True nothing will be reversed. Defaults to False.

        Raises:
            DeephyperRuntimeError: [description]

        Returns:
            tuple: (y_true, y_pred)
        """
        if not (dataset == "valid" or dataset == "train"):
            raise DeephyperRuntimeError(
                "dataset parameter should be equal to: 'valid' or 'train'"
            )

        if dataset == "valid":
            y_pred = self.model.predict(
                self.dataset_valid, steps=self.valid_steps_per_epoch
            )
        else:  # dataset == 'train'
            y_pred = self.model.predict(
                self.dataset_train, steps=self.train_steps_per_epoch
            )

        if (
            self.preprocessing_func
            and not keep_normalize
            and not self.data_config_type == "gen"
        ):
            if dataset == "valid":
                data_X, data_Y = self.valid_X, self.valid_Y
            else:  # dataset == 'train'
                data_X, data_Y = self.train_X, self.train_Y
            val_pred = np.concatenate((*data_X, y_pred), axis=1)
            val_orig = np.concatenate((*data_X, data_Y), axis=1)
            val_pred_trans = self.preprocessor.inverse_transform(val_pred)
            val_orig_trans = self.preprocessor.inverse_transform(val_orig)
            y_orig = val_orig_trans[:, -np.shape(data_Y)[1] :]
            y_pred = val_pred_trans[:, -np.shape(data_Y)[1] :]
        else:
            if self.data_config_type == "ndarray":
                y_orig = self.valid_Y if dataset == "valid" else self.train_Y
            else:
                gen = self.valid_gen() if dataset == "valid" else self.train_gen()
                y_orig = np.array([e[-1] for e in gen])

        return y_orig, y_pred

    def evaluate(self, dataset="train"):
        """Evaluate the performance of your model for the same configuration.

        Args:
            dataset (str, optional): must be "train" or "valid". If "train" then metrics will be evaluated on the training dataset. If "valid" then metrics will be evaluated on the "validation" dataset. Defaults to 'train'.

        Returns:
            list: a list of scalar values corresponding do config loss & metrics.
        """
        if dataset == "train":
            return self.model.evaluate(
                self.dataset_train, steps=self.train_steps_per_epoch
            )
        else:
            return self.model.evaluate(
                self.dataset_valid, steps=self.valid_steps_per_epoch
            )

    def train(
        self, num_epochs: int = None, with_pred: bool = False, last_only: bool = False
    ):
        """Train the model.

        Args:
            num_epochs (int, optional): override the num_epochs passed to init the Trainer. Defaults to None, will use the num_epochs passed to init the Trainer.
            with_pred (bool, optional): will compute a prediction after the training and will add ('y_true', 'y_pred') to the output history. Defaults to False, will skip it (use it to save compute time).
            last_only (bool, optional): will compute metrics after the last epoch only. Defaults to False, will compute metrics after each training epoch (use it to save compute time).

        Raises:
            DeephyperRuntimeError: raised when the ``num_epochs < 0``.

        Returns:
            dict: a dictionnary corresponding to the training.
        """
        num_epochs = self.num_epochs if num_epochs is None else num_epochs

        self.init_history()

        if num_epochs > 0:

            time_start_training = time.time()  # TIMING

            if not last_only:
                logger.info(
                    "Trainer is computing metrics on validation after each training epoch."
                )
                history = self.model.fit(
                    self.dataset_train,
                    verbose=self.verbose,
                    epochs=num_epochs,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch,
                    class_weight=self.class_weights,
                )
            else:
                logger.info(
                    "Trainer is computing metrics on validation after the last training epoch."
                )
                if num_epochs > 1:
                    self.model.fit(
                        self.dataset_train,
                        verbose=self.verbose,
                        epochs=num_epochs - 1,
                        steps_per_epoch=self.train_steps_per_epoch,
                        callbacks=self.callbacks,
                        class_weight=self.class_weights,
                    )
                history = self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    verbose=self.verbose,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch,
                    class_weight=self.class_weights,
                )

            time_end_training = time.time()  # TIMING
            self.train_history["training_time"] = (
                time_end_training - time_start_training
            )

            self.train_history.update(history.history)

        elif num_epochs < 0:

            raise DeephyperRuntimeError(
                f"Trainer: number of epochs should be >= 0: {num_epochs}"
            )

        if with_pred:
            time_start_predict = time.time()
            y_true, y_pred = self.predict(dataset="valid")
            time_end_predict = time.time()
            self.train_history["val_predict_time"] = (
                time_end_predict - time_start_predict
            )

            self.train_history["y_true"] = y_true
            self.train_history["y_pred"] = y_pred

        return self.train_history

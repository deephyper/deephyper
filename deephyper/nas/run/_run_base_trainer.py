"""The :func:`deephyper.nas.run.alpha.run` function is used to evaluate a deep neural network by loading the data, building the model, training the model and returning a scalar value corresponding to the objective defined in the used :class:`deephyper.problem.NaProblem`.
"""
import os
import traceback
import logging

import numpy as np
import tensorflow as tf
from deephyper.keras.callbacks import import_callback
from deephyper.nas.run._util import (
    compute_objective,
    load_config,
    preproc_trainer,
    setup_data,
    get_search_space,
    default_callbacks_config,
    HistorySaver,
)
from deephyper.nas.trainer import BaseTrainer

logger = logging.getLogger(__name__)


def run_base_trainer(config):

    tf.keras.backend.clear_session()
    # tf.config.optimizer.set_jit(True)

    # setup history saver
    if config.get("log_dir") is None:
        config["log_dir"] = "."

    save_dir = os.path.join(config["log_dir"], "save")
    saver = HistorySaver(config, save_dir)
    saver.write_config()
    saver.write_model(None)

    # GPU Configuration if available
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        logger.info("error memory growth for GPU device")

    # Threading configuration
    if (
        len(physical_devices) == 0
        and os.environ.get("OMP_NUM_THREADS", None) is not None
    ):
        logger.info(f"OMP_NUM_THREADS is {os.environ.get('OMP_NUM_THREADS')}")
        num_intra = int(os.environ.get("OMP_NUM_THREADS"))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(num_intra)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:  # Session already initialized
            pass
        tf.config.set_soft_device_placement(True)

    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    input_shape, output_shape = setup_data(config)

    search_space = get_search_space(config, input_shape, output_shape, seed=seed)

    model_created = False
    try:
        model = search_space.sample(config["arch_seq"])
        model_created = True
    except:
        logger.info("Error: Model creation failed...")
        logger.info(traceback.format_exc())

    if model_created:

        # Setup callbacks
        callbacks = []
        cb_requires_valid = False  # Callbacks requires validation data
        callbacks_config = config["hyperparameters"].get("callbacks")
        if callbacks_config is not None:
            for cb_name, cb_conf in callbacks_config.items():
                if cb_name in default_callbacks_config:
                    default_callbacks_config[cb_name].update(cb_conf)

                    # Special dynamic parameters for callbacks
                    if cb_name == "ModelCheckpoint":
                        default_callbacks_config[cb_name]["filepath"] = saver.model_path

                    # replace patience hyperparameter
                    if "patience" in default_callbacks_config[cb_name]:
                        patience = config["hyperparameters"].get(f"patience_{cb_name}")
                        if patience is not None:
                            default_callbacks_config[cb_name]["patience"] = patience

                    # Import and create corresponding callback
                    Callback = import_callback(cb_name)
                    callbacks.append(Callback(**default_callbacks_config[cb_name]))

                    if cb_name in ["EarlyStopping"]:
                        cb_requires_valid = "val" in cb_conf["monitor"].split("_")
                else:
                    logger.error(f"'{cb_name}' is not an accepted callback!")

        trainer = BaseTrainer(config=config, model=model)
        trainer.callbacks.extend(callbacks)

        last_only, with_pred = preproc_trainer(config)
        last_only = last_only and not cb_requires_valid

        history = trainer.train(with_pred=with_pred, last_only=last_only)

        # save history
        saver.write_history(history)

        result = compute_objective(config["objective"], history)
    else:
        # penalising actions if model cannot be created
        logger.info("Model could not be created returning -Inf!")
        result = -float("inf")

    if np.isnan(result):
        logger.info("Computed objective is NaN returning -Inf instead!")
        result = -float("inf")

    return result

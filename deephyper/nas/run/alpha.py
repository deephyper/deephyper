"""The :func:`deephyper.nas.run.alpha.run` function is used to evaluate a deep neural network by loading the data, building the model, training the model and returning a scalar value corresponding to the objective defined in the used :class:`deephyper.problem.NaProblem`.
"""
import os
import traceback

import numpy as np
import tensorflow as tf
from deephyper.contrib.callbacks import import_callback
from deephyper.nas.run.util import (
    compute_objective,
    load_config,
    preproc_trainer,
    save_history,
    setup_data,
    setup_search_space,
    default_callbacks_config,
)
from deephyper.nas.trainer.train_valid import TrainerTrainValid
from deephyper.search import util

logger = util.conf_logger("deephyper.search.nas.run")


def run(config: dict) -> float:
    # Threading configuration
    if os.environ.get("OMP_NUM_THREADS", None) is not None:
        logger.debug(f"OMP_NUM_THREADS is {os.environ.get('OMP_NUM_THREADS')}")
        num_intra = int(os.environ.get("OMP_NUM_THREADS"))
        tf.config.threading.set_intra_op_parallelism_threads(num_intra)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    input_shape, output_shape = setup_data(config)

    search_space = setup_search_space(config, input_shape, output_shape, seed=seed)

    model_created = False
    try:
        model = search_space.create_model()
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
                        default_callbacks_config[cb_name][
                            "filepath"
                        ] = f'best_model_{config["id"]}.h5'

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

        trainer = TrainerTrainValid(config=config, model=model)
        trainer.callbacks.extend(callbacks)

        last_only, with_pred = preproc_trainer(config)
        last_only = last_only and not cb_requires_valid

        history = trainer.train(with_pred=with_pred, last_only=last_only)

        # save history
        save_history(config.get("log_dir", None), history, config)

        result = compute_objective(config["objective"], history)
    else:
        # penalising actions if model cannot be created
        result = -1
    if result < -10 or np.isnan(result):
        result = -10
    return result
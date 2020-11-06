import os
import traceback

import numpy as np
import tensorflow as tf
from deephyper.contrib.callbacks import import_callback
from deephyper.search import util

from ..trainer.train_valid import TrainerTrainValid
from .util import (
    compute_objective,
    load_config,
    preproc_trainer,
    setup_data,
    setup_search_space,
)

logger = util.conf_logger("deephyper.search.nas.run")

# Default callbacks parameters
default_callbacks_config = {
    "EarlyStopping": dict(
        monitor="val_loss", min_delta=0, mode="min", verbose=0, patience=0
    ),
    "ModelCheckpoint": dict(
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
        filepath="model.h5",
        save_weights_only=False,
    ),
    "TensorBoard": dict(
        log_dir="",
        histogram_freq=0,
        batch_size=32,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq="epoch",
    ),
    "CSVLogger": dict(filename="training.csv", append=True),
    "CSVExtendedLogger": dict(filename="training.csv", append=True),
    "TimeStopping": dict(),
}


def run(config):
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

        result = compute_objective(config["objective"], history)
    else:
        # penalising actions if model cannot be created
        result = -1
    if result < -10:
        result = -10
    return result

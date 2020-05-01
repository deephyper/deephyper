import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd

from ....search import util
from ..trainer.horovod_trainer import HorovodTrainerTrainValid
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
    "TensorBoard": dict(
        log_dir="",
        histogram_freq=0,
        batch_size=32,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq="epoch",
    ),
    "CSVLogger": dict(
        filename="training.csv",
        append=True
    )
}
# Name of Callbacks reserved for root node
hvd_root_cb = ["ModelCheckpoint", "Tensorboard", "CSVLogger"]


def run(config):
    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        if tf.__version__ == "1.13.1":
            tf.random.set_random_seed(seed)
        else:
            tf.compat.v2.random.set_seed(seed)

    load_config(config)

    input_shape, output_shape = setup_data(config)

    search_space = setup_search_space(config, input_shape, output_shape, seed=seed)

    # Initialize Horovod
    hvd.init()

    model_created = False
    try:
        model = search_space.create_model()
        model_created = True
    except:
        logger.info("Error: Model creation failed...")
        logger.info(traceback.format_exc())

    if model_created:

        # Setup callbacks only
        callbacks = []
        cb_requires_valid = False  # Callbacks requires validation data
        callbacks_config = config["hyperparameters"].get("callbacks")
        if callbacks_config is not None:
            for cb_name, cb_conf in callbacks_config.items():
                if cb_name in default_callbacks_config:
                    # cb_bame in hvd_root_cb implies hvd.rank() == 0
                    if not(cb_name in hvd_root_cb) or hvd.rank() == 0:
                        default_callbacks_config[cb_name].update(cb_conf)

                        # Special dynamic parameters for callbacks
                        if cb_name == "ModelCheckpoint":
                            default_callbacks_config[cb_name][
                                "filepath"
                            ] = f'best_model_{config["id"]}.h5'

                        # Import and create corresponding callback
                        Callback = getattr(keras.callbacks, cb_name)
                        callbacks.append(Callback(**default_callbacks_config[cb_name]))

                        if cb_name in ["EarlyStopping"]:
                            cb_requires_valid = "val" in cb_conf["monitor"].split("_")
                else:
                    logger.error(f"'{cb_name}' is not an accepted callback!")

        trainer = HorovodTrainerTrainValid(config=config, model=model)
        callbacks.append(
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        )
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

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
)
from deephyper.nas.trainer.train_valid import TrainerTrainValid
from deephyper.search import util
import deephyper.nas.arch as a

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
    "ReduceLROnPlateau":dict(monitor="val_loss", mode="auto", verbose=0, patience=5)
}


def run(config):
    distributed_strategy = tf.distribute.MirroredStrategy()
    n_replicas = distributed_strategy.num_replicas_in_sync

    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    # Scale batch size and learning rate according to the number of ranks
    batch_size = config[a.hyperparameters][a.batch_size] * n_replicas
    learning_rate = config[a.hyperparameters][a.learning_rate]
    logger.info(
        f"Scaled: 'batch_size' from {config[a.hyperparameters][a.batch_size]} to {batch_size} "
    )
    logger.info(
        f"Scaled: 'learning_rate' from {config[a.hyperparameters][a.learning_rate]} to {learning_rate} "
    )
    config[a.hyperparameters][a.batch_size] = batch_size
    config[a.hyperparameters][a.learning_rate] = learning_rate

    input_shape, output_shape = setup_data(config)

    search_space = setup_search_space(config, input_shape, output_shape, seed=seed)

    model_created = False
    with distributed_strategy.scope():
        try:
            model = search_space.create_model()
            model_created = True
        except:
            logger.info("Error: Model creation failed...")
            logger.info(traceback.format_exc())
        else:
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

    if model_created:
            history = trainer.train(with_pred=with_pred, last_only=last_only)

            # save history
            save_history(config.get("log_dir", None), history, config)

            result = compute_objective(config["objective"], history)
    else:
        # penalising actions if model cannot be created
        result = -1
    if result < -10:
        result = -10
    return result
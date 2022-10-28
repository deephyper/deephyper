"""The :func:`deephyper.nas.run.horovod.run` function is used to evaluate a deep neural network by enabling data-parallelism with Horovod to the :func:`deephyper.nas.run.alpha.run` function. This function will automatically apply the linear scaling rule to the learning rate and batch size given the current number of ranks (i.e., the initial learning rate and batch size are scaled by the number of ranks).
"""
import os
import traceback
import logging

import numpy as np
import tensorflow as tf
from deephyper.keras.callbacks import import_callback

import horovod.tensorflow.keras as hvd

import deephyper.nas.trainer._arch as a
from deephyper.nas.trainer import HorovodTrainer
from deephyper.nas.run._util import (
    compute_objective,
    load_config,
    preproc_trainer,
    save_history,
    setup_data,
    get_search_space,
)

logger = logging.getLogger(__name__)

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
    "ReduceLROnPlateau": dict(patience=5, verbose=0),
}
# Name of Callbacks reserved for root node
hvd_root_cb = ["ModelCheckpoint", "Tensorboard", "CSVLogger", "CSVExtendedLogger"]


def run_horovod(config: dict) -> float:
    hvd.init()

    # Threading configuration
    if os.environ.get("OMP_NUM_THREADS", None) is not None:
        logger.debug(f"OMP_NUM_THREADS is {os.environ.get('OMP_NUM_THREADS')}")
        num_intra = int(os.environ.get("OMP_NUM_THREADS"))
        tf.config.threading.set_intra_op_parallelism_threads(num_intra)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[hvd.rank()]

    config["seed"]
    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    # Scale batch size and learning rate according to the number of ranks
    initial_lr = config[a.hyperparameters][a.learning_rate]
    batch_size = config[a.hyperparameters][a.batch_size] * hvd.size()
    learning_rate = config[a.hyperparameters][a.learning_rate] * hvd.size()
    logger.info(
        f"Scaled: 'batch_size' from {config[a.hyperparameters][a.batch_size]} to {batch_size} "
    )
    logger.info(
        f"Scaled: 'learning_rate' from {config[a.hyperparameters][a.learning_rate]} to {learning_rate} "
    )
    config[a.hyperparameters][a.batch_size] = batch_size
    config[a.hyperparameters][a.learning_rate] = learning_rate

    input_shape, output_shape = setup_data(config)

    search_space = get_search_space(config, input_shape, output_shape, seed=seed)

    # Initialize Horovod

    model_created = False
    try:
        model = search_space.sample(config["arch_seq"])
        model_created = True
    except Exception:
        logger.info("Error: Model creation failed...")
        logger.info(traceback.format_exc())

    if model_created:

        # Setup callbacks only
        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),
            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            # !initial_lr argument is not available in horovod==0.19.0
            hvd.callbacks.LearningRateWarmupCallback(
                warmup_epochs=5, verbose=0, initial_lr=initial_lr
            ),
        ]

        cb_requires_valid = False  # Callbacks requires validation data
        callbacks_config = config[a.hyperparameters].get(a.callbacks, {})
        if callbacks_config is not None:
            for cb_name, cb_conf in callbacks_config.items():
                if cb_name in default_callbacks_config:
                    # cb_bame in hvd_root_cb implies hvd.rank() == 0
                    if not (cb_name in hvd_root_cb) or hvd.rank() == 0:
                        default_callbacks_config[cb_name].update(cb_conf)

                        # Import and create corresponding callback
                        Callback = import_callback(cb_name)
                        callbacks.append(Callback(**default_callbacks_config[cb_name]))

                        if cb_name in ["EarlyStopping"]:
                            cb_requires_valid = "val" in cb_conf["monitor"].split("_")
                else:
                    logger.error(f"'{cb_name}' is not an accepted callback!")

        trainer = HorovodTrainer(config=config, model=model)

        trainer.callbacks.extend(callbacks)

        last_only, with_pred = preproc_trainer(config)
        last_only = last_only and not cb_requires_valid

        history = trainer.train(with_pred=with_pred, last_only=last_only)

        # save history
        if hvd.rank() == 0:
            save_history(config.get("log_dir", None), history, config)

        result = compute_objective(config["objective"], history)
    else:
        # penalising actions if model cannot be created
        result = -1
    if result < -10:
        result = -10
    return result

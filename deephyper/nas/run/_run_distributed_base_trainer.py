"""The :func:`deephyper.nas.run.tf_distributed.run` function is used to deploy a data-distributed training (on a single node) with ``tensorflow.distribute.MirroredStrategy``. It follows the same training pipeline as :func:`deephyper.nas.run.alpha.run`. Two hyperparameters arguments can be used to activate or deactivate the linear scaling rule (aka ``lsr``) for the learning rate and batch size, respectively:

.. code-block:: python

    Problem.hyperparameters(
        ...
        lsr_batch_size=True,
        lsr_learning_rate=True,
        warmup_lr=True,
        warmup_epochs=5
        ...
    )

The hyperparameters of the form ``patience_{Callback}`` such as ``patience_EarlyStopping`` and ``patience_ReduceLROnPlateau`` are valid when the corresponding callback is declared:

.. code-block:: python

    Problem.hyperparameters(
        ...
        patience_ReduceLROnPlateau=5,
        patience_EarlyStopping=10,
        callbacks=dict(
            ReduceLROnPlateau=dict(monitor="val_r2", mode="max", verbose=0),
            EarlyStopping=dict(monitor="val_r2", min_delta=0, mode="max", verbose=0),
        ),
        ...
    )
"""
import traceback
import logging

import numpy as np
import tensorflow as tf
from deephyper.keras.callbacks import import_callback
from deephyper.keras.callbacks import LearningRateWarmupCallback
from deephyper.nas.run._util import (
    compute_objective,
    load_config,
    preproc_trainer,
    save_history,
    setup_data,
    get_search_space,
    default_callbacks_config,
)
from deephyper.nas.trainer import BaseTrainer
import deephyper.nas.trainer._arch as a

logger = logging.getLogger(__name__)


def run_distributed_base_trainer(config):

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    distributed_strategy = tf.distribute.MirroredStrategy()
    n_replicas = distributed_strategy.num_replicas_in_sync

    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    # Scale batch size and learning rate according to the number of ranks
    initial_lr = config[a.hyperparameters][a.learning_rate]
    if config[a.hyperparameters].get("lsr_batch_size"):
        batch_size = config[a.hyperparameters][a.batch_size] * n_replicas
    else:
        batch_size = config[a.hyperparameters][a.batch_size]
    if config[a.hyperparameters].get("lsr_learning_rate"):
        learning_rate = config[a.hyperparameters][a.learning_rate] * n_replicas
    else:
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

    search_space = get_search_space(config, input_shape, output_shape, seed=seed)

    model_created = False
    with distributed_strategy.scope():
        try:
            model = search_space.sample(config["arch_seq"])
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

                        # replace patience hyperparameter
                        if "patience" in default_callbacks_config[cb_name]:
                            patience = config["hyperparameters"].get(
                                f"patience_{cb_name}"
                            )
                            if patience is not None:
                                default_callbacks_config[cb_name]["patience"] = patience

                        # Import and create corresponding callback
                        Callback = import_callback(cb_name)
                        callbacks.append(Callback(**default_callbacks_config[cb_name]))

                        if cb_name in ["EarlyStopping"]:
                            cb_requires_valid = "val" in cb_conf["monitor"].split("_")
                    else:
                        logger.error(f"'{cb_name}' is not an accepted callback!")

            # WarmupLR
            if config[a.hyperparameters].get("warmup_lr"):
                warmup_epochs = config[a.hyperparameters].get("warmup_epochs", 5)
                callbacks.append(
                    LearningRateWarmupCallback(
                        n_replicas=n_replicas,
                        warmup_epochs=warmup_epochs,
                        verbose=0,
                        initial_lr=initial_lr,
                    )
                )

            trainer = BaseTrainer(config=config, model=model)
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
    if result < -10 or np.isnan(result):
        result = -10

    return result

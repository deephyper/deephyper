import json
import traceback
from time import time
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from deephyper.evaluator import Encoder
from deephyper.search import util
from deephyper.nas.run.util import (
    load_config,
    setup_data,
    setup_search_space,
    compute_objective,
)
from deephyper.nas.trainer.train_valid import TrainerTrainValid
from deephyper.contrib.callbacks.beholder import BeholderCB

logger = util.conf_logger(__name__)

CB_CONFIG = {
    "ModelCheckpoint": dict(
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
        filepath="best_model.h5",
        save_weights_only=False,
    ),
    "EarlyStopping": dict(monitor="val_loss", mode="min", verbose=1, patience=50),
    "TensorBoard": dict(
        log_dir="",
        histogram_freq=0,
        batch_size=32,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq="epoch",
        beholder=False,
    ),
}


def train(config):
    seed = config["seed"]
    repeat = config["post_train"]["repeat"]
    if seed is not None:
        np.random.seed(seed)
        # must be between (0, 2**32-1)
        seeds = [np.random.randint(0, 2 ** 32 - 1) for _ in range(repeat)]

    for rep in range(repeat):
        tf.keras.backend.clear_session()

        default_callbacks_config = copy.deepcopy(CB_CONFIG)
        if seed is not None:
            np.random.seed(seeds[rep])
            tf.random.set_seed(seeds[rep])

        logger.info(f"Training replica {rep+1}")
        # Pre-settings: particularly import for BeholderCB to work
        sess = tf.Session()
        K.set_session(sess)

        # override hyperparameters with post_train hyperparameters
        keys = filter(
            lambda k: k in config["hyperparameters"], config["post_train"].keys()
        )
        for k in keys:
            config["hyperparameters"][k] = config["post_train"][k]

        load_config(config)

        input_shape, output_shape = setup_data(config)

        search_space = setup_search_space(config, input_shape, output_shape, seed=seed)
        search_space.draw_graphviz(f'model_{config["id"]}.dot')
        logger.info("Model operations set.")

        model_created = False
        try:
            model = search_space.create_model()
            model_created = True
        except:
            model_created = False
            logger.info("Error: Model creation failed...")
            logger.info(traceback.format_exc())

        if model_created:
            # model.load_weights(default_cfg['model_checkpoint']['filepath'])

            # Setup callbacks
            callbacks = []
            callbacks_config = config["post_train"].get("callbacks")
            if callbacks_config is not None:
                for cb_name, cb_conf in callbacks_config.items():
                    if cb_name in default_callbacks_config:
                        default_callbacks_config[cb_name].update(cb_conf)

                        if cb_name == "ModelCheckpoint":
                            default_callbacks_config[cb_name][
                                "filepath"
                            ] = f'best_model_id{config["id"]}_r{rep}.h5'
                        elif cb_name == "TensorBoard":
                            if default_callbacks_config[cb_name]["beholder"]:
                                callbacks.append(
                                    BeholderCB(
                                        logdir=default_callbacks_config[cb_name][
                                            "log_dir"
                                        ],
                                        sess=sess,
                                    )
                                )
                            default_callbacks_config[cb_name].pop("beholder")

                        Callback = getattr(keras.callbacks, cb_name)
                        callbacks.append(Callback(**default_callbacks_config[cb_name]))

                        logger.info(
                            f"Adding new callback {type(Callback).__name__} with config: {default_callbacks_config[cb_name]}!"
                        )

                    else:
                        logger.error(f"'{cb_name}' is not an accepted callback!")

            trainer = TrainerTrainValid(config=config, model=model)
            trainer.callbacks.extend(callbacks)

            json_fname = f'post_training_hist_{config["id"]}.json'
            # to log the number of trainable parameters before running training
            trainer.init_history()
            try:
                with open(json_fname, "r") as f:
                    fhist = json.load(f)
            except FileNotFoundError:
                fhist = trainer.train_history
                for k, v in fhist.items():
                    fhist[k] = [v]
                with open(json_fname, "w") as f:
                    json.dump(fhist, f, cls=Encoder)

            hist = trainer.train(with_pred=False, last_only=False)

            # Timing of prediction for validation dataset
            t = time()  # ! TIMING - START
            trainer.predict(dataset="valid")
            hist["val_predict_time"] = time() - t  # ! TIMING - END

            for k, v in hist.items():
                fhist[k] = fhist.get(k, [])
                fhist[k].append(v)

            with open(json_fname, "w") as f:
                json.dump(fhist, f, cls=Encoder)

        return model

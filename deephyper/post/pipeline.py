import json
import traceback
from time import time

import numpy as np
from tensorflow import keras

from deephyper.evaluator import Encoder
from deephyper.search import util
from deephyper.search.nas.model.run.util import load_config, setup_data, setup_structure, compute_objective
from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid

logger = util.conf_logger(__name__)

default_callbacks_config = {
    'ModelCheckpoint': dict(
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1,
        filepath="best_model.h5",
        save_weights_only=False
    ),
    'EarlyStopping': dict(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=50
    ),
    'TensorBoard': dict(
        logdir='',
        histogram_freq=0,
        batch_size=32,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq='epoch'
    )
}


def train(config):

    # Pre-settings

    # override hyperparameters with post_train hyperparameters
    keys = filter(lambda k: k in config['hyperparameters'],
                  config['post_train'].keys())
    for k in keys:
        config['hyperparameters'][k] = config['post_train'][k]

    load_config(config)

    input_shape, output_shape = setup_data(config)

    architecture = setup_structure(config, input_shape, output_shape)
    architecture.draw_graphviz(f'structure_{config["id"]}.dot')
    logger.info('Model operations set.')

    model_created = False
    try:
        model = architecture.create_model()
        model_created = True
    except:
        model_created = False
        logger.info('Error: Model creation failed...')
        logger.info(traceback.format_exc())



    if model_created:
        # model.load_weights(default_cfg['model_checkpoint']['filepath'])

        # Setup callbacks
        callbacks = []
        callbacks_config = config['post_train'].get('callbacks')
        if callbacks_config is not None:
            for cb_name, cb_conf in callbacks_config.items():
                if cb_name in default_callbacks_config:
                    default_callbacks_config[cb_name].update(cb_conf)

                    if cb_name == 'ModelCheckpoint':
                        default_callbacks_config[cb_name]['filepath'] =  f'best_model_{config["id"]}.h5'

                    Callback = getattr(keras.callbacks, cb_name)
                    callbacks.append(Callback(**default_callbacks_config[cb_name]))

                    logger.info(f'Adding new callback {type(Callback).__name__} with config: {default_callbacks_config[cb_name]}!')

                else:
                    logger.error(f"'{cb_name}' is not an accepted callback!")

        trainer = TrainerTrainValid(config=config, model=model)
        trainer.callbacks.extend(callbacks)

        json_fname = f'post_training_hist_{config["id"]}.json'
        # to log the number of trainable parameters before running training
        trainer.init_history()
        with open(json_fname, 'w') as f:
            json.dump(trainer.train_history, f, cls=Encoder)


        hist = trainer.train(with_pred=False, last_only=False)

        # Timing of prediction for validation dataset
        t = time()  # ! TIMING - START
        trainer.predict(dataset='valid')
        hist['val_predict_time'] = time() - t  # ! TIMING - END

        with open(json_fname, 'w') as f:
            json.dump(hist, f, cls=Encoder)

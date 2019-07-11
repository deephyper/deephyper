import json
import sys
import traceback
from time import time

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from deephyper.evaluator import Encoder
from deephyper.search import util
from deephyper.search.nas.model.run.util import load_config, setup_data, setup_structure, compute_objective
from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid

logger = util.conf_logger(__name__)

default_cfg = {
    'model_checkpoint': dict(
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    'early_stopping': dict(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=50
    )
}


def train(config):

    # Pre-settings

    # override hyperparameters with post_train hyperparameters
    keys = filter(lambda k: k in config['hyperparameters'],
                  config['post_train'].keys())
    for k in keys:
        config['hyperparameters'][k] = config['post_train'][k]

    # override default callbacks configs with post_train configs
    keys = filter(lambda k: k in default_cfg,
                  config['post_train'].keys())
    for k in keys:
        default_cfg[k] = config['post_train'][k]

    load_config(config)

    input_shape, output_shape = setup_data(config)

    structure = setup_structure(config, input_shape, output_shape)
    structure.draw_graphviz(f'structure_{config["id"]}.dot')
    logger.info('Model operations set.')

    model_created = False
    try:
        model = structure.create_model()
        model_created = True
    except:
        model_created = False
        logger.info('Error: Model creation failed...')
        logger.info(traceback.format_exc())

    if model_created:
        trainer = TrainerTrainValid(config=config, model=model)

    if model_created:
        trainer.callbacks.append(EarlyStopping(
            **default_cfg['early_stopping']))
        trainer.callbacks.append(ModelCheckpoint(
            f'best_model_{config["id"]}.h5',
            **default_cfg['model_checkpoint']))

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

import json
import traceback
from time import time
import sys

import numpy as np
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from deephyper.evaluator import Encoder
from deephyper.search import util
from deephyper.search.nas.model.trainer.classifier_train_valid import \
    TrainerClassifierTrainValid
from deephyper.search.nas.model.trainer.regressor_train_valid import \
    TrainerRegressorTrainValid

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
    keys = filter(lambda k: k in config['hyperparameters'],
                  config['post_train'].keys())
    for k in keys:
        config['hyperparameters'][k] = config['post_train'][k]

    keys = filter(lambda k: k in default_cfg,
                  config['post_train'].keys())
    for k in keys:
        default_cfg[k] = config['post_train'][k]

    load_data = config['load_data']['func']

    # load functions
    load_data = util.load_attr_from(config['load_data']['func'])
    config['load_data']['func'] = load_data
    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    logger.info('Loading data...')
    # Loading data
    kwargs = config['load_data'].get('kwargs')
    data = load_data() if kwargs is None else load_data(**kwargs)
    logger.info('Data loaded!')

    # Set data shape
    if type(data) is tuple:
        if len(data) != 2:
            raise RuntimeError(
                f'Loaded data are tuple, should ((training_input, training_output), (validation_input, validation_output)) but length=={len(data)}')
        (t_X, t_y), (v_X, v_y) = data
        if type(t_X) is np.ndarray and type(t_y) is np.ndarray and \
                type(v_X) is np.ndarray and type(v_y) is np.ndarray:
            input_shape = np.shape(t_X)[1:]
        elif type(t_X) is list and type(t_y) is np.ndarray and \
                type(v_X) is list and type(v_y) is np.ndarray:
            # interested in shape of data not in length
            input_shape = [np.shape(itX)[1:] for itX in t_X]
        else:
            raise RuntimeError(
                f'Data returned by load_data function are of a wrong type: type(t_X)=={type(t_X)},  type(t_y)=={type(t_y)}, type(v_X)=={type(v_X)}, type(v_y)=={type(v_y)}')
        output_shape = np.shape(t_y)[1:]
        config['data'] = {
            'train_X': t_X,
            'train_Y': t_y,
            'valid_X': v_X,
            'valid_Y': v_y
        }
    elif type(data) is dict:
        config['data'] = data
        input_shape = [data['shapes'][0][f'input_{i}']
                       for i in range(len(data['shapes'][0]))]
        output_shape = data['shapes'][1]
    else:
        raise RuntimeError(
            f'Data returned by load_data function are of an unsupported type: {type(data)}')

    structure = config['create_structure']['func'](
        input_shape, output_shape, **config['create_structure']['kwargs'])

    arch_seq = config['arch_seq']

    structure.set_ops(arch_seq)
    structure.draw_graphviz(f'structure_{config["id"]}.dot')
    logger.info('Model operations set.')

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
    else:
        config['preprocessing'] = None

    model_created = False
    if config['regression']:
        try:
            model = structure.create_model()
            model_created = True
        except:
            model_created = False
            logger.error('Model creation failed...')
            logger.error(traceback.format_exc())
        if model_created:
            trainer = TrainerRegressorTrainValid(config=config, model=model)
    else:
        try:
            model = structure.create_model(
                activation='softmax')
            model_created = True
        except:
            model_created = False
            logger.error('Model creation failed...')
            logger.error(traceback.format_exc())
        if model_created:
            trainer = TrainerClassifierTrainValid(config=config, model=model)

    if model_created:
        trainer.callbacks.append(EarlyStopping(
            **default_cfg['early_stopping']))
        trainer.callbacks.append(ModelCheckpoint(
            f'best_model_{config["id"]}.h5',
            **default_cfg['model_checkpoint']))

        t = time()  # ! TIMING - START
        trainer.post_train()
        hist = trainer.post_train_history
        hist['training_time'] = time() - t  # ! TIMING - END

        # Timing of prediction for validation dataset
        t = time()  # ! TIMING - START
        trainer.predict()
        hist['val_predict_time'] = time() - t  # ! TIMING - END

        hist['n_parameters'] = model.count_params()

        print(hist)
        with open(f'post_training_hist_{config["id"]}.json', 'w') as f:
            json.dump(hist, f, cls=Encoder)
        return min(hist['val_loss'])
    else:
        return sys.float_info.max

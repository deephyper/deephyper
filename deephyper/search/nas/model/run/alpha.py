from random import random
import traceback

import numpy as np
from tensorflow import keras

from deephyper.search import util
from deephyper.search.nas.model.trainer.classifier_train_valid import \
    TrainerClassifierTrainValid
from deephyper.search.nas.model.trainer.regressor_train_valid import \
    TrainerRegressorTrainValid

logger = util.conf_logger('deephyper.search.nas.run')

def run(config):
    # load functions
    load_data = util.load_attr_from(config['load_data']['func'])
    config['load_data']['func'] = load_data
    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    # Loading data
    kwargs = config['load_data'].get('kwargs')
    data = load_data() if kwargs is None else load_data(**kwargs)
    logger.info(f'Data loaded with kwargs: {kwargs}')

    # Set data shape
    if type(data) is tuple:
        if len(data) != 2:
            raise RuntimeError(f'Loaded data are tuple, should ((training_input, training_output), (validation_input, validation_output)) but length=={len(data)}')
        (t_X, t_y), (v_X, v_y) = data
        if type(t_X) is np.ndarray and  type(t_y) is np.ndarray and \
            type(v_X) is np.ndarray and type(v_y) is np.ndarray:
            input_shape = np.shape(t_X)[1:]
        elif type(t_X) is list and type(t_y) is np.ndarray and \
            type(v_X) is list and type(v_y) is np.ndarray:
            input_shape = [np.shape(itX)[1:] for itX in t_X] # interested in shape of data not in length
        else:
            raise RuntimeError(f'Data returned by load_data function are of a wrong type: type(t_X)=={type(t_X)},  type(t_y)=={type(t_y)}, type(v_X)=={type(v_X)}, type(v_y)=={type(v_y)}')
        output_shape = np.shape(t_y)[1:]
        config['data'] = {
            'train_X': t_X,
            'train_Y': t_y,
            'valid_X': v_X,
            'valid_Y': v_y
        }
    elif type(data) is dict:
        config['data'] = data
        input_shape = [data['shapes'][0][f'input_{i}'] for i in range(len(data['shapes'][0]))]
        output_shape = data['shapes'][1]
    else:
        raise RuntimeError(f'Data returned by load_data function are of an unsupported type: {type(data)}')

    logger.info(f'input_shape: {input_shape}')
    logger.info(f'output_shape: {output_shape}')

    cs_kwargs = config['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = config['create_structure']['func'](input_shape, output_shape)
    else:
        structure = config['create_structure']['func'](input_shape, output_shape, **cs_kwargs)

    arch_seq = config['arch_seq']

    logger.info(f'actions list: {arch_seq}')

    structure.set_ops(arch_seq)

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
            logger.info('Error: Model creation failed...')
            logger.info(traceback.format_exc())
        if model_created:
            trainer = TrainerRegressorTrainValid(config=config, model=model)
    else:
        try:
            model = structure.create_model(activation='softmax')
            model_created = True
        except:
            model_created = False
            logger.info('Error: Model creation failed...')
            logger.info(traceback.format_exc())
        if model_created:
            trainer = TrainerClassifierTrainValid(config=config, model=model)

    if model_created:
        # 0 < reward regression < 105
        # 0 < reward classification < 100
        res = trainer.train()
        if config['regression']:
            if res < np.finfo('float32').min:
                res = np.finfo('float32').min
            res = - np.log(res) + np.log(np.finfo('float32').max)
        result = res
    else:
        # penalising actions if model cannot be created
        result = -1
    return result

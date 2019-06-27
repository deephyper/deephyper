from random import random
import traceback

import numpy as np
from tensorflow import keras

from deephyper.search import util

from deephyper.search.nas.model.trainer.train_valid import TrainerTrainValid

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

    logger.info(f'input_shape: {input_shape}')
    logger.info(f'output_shape: {output_shape}')

    cs_kwargs = config['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = config['create_structure']['func'](
            input_shape, output_shape)
    else:
        structure = config['create_structure']['func'](
            input_shape, output_shape, **cs_kwargs)

    arch_seq = config['arch_seq']

    logger.info(f'actions list: {arch_seq}')

    structure.set_ops(arch_seq)

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
    else:
        config['preprocessing'] = None

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

        last_only = '__last' in config['objective'] \
            or '__last' in config['objective'].__name__
        with_pred = not type(config['objective']) is str \
            and 'with_pred' in config['objective'].__name__

        history = trainer.train(with_pred=with_pred, last_only=last_only)

        result = compute_objective(config['objective'], history)
    else:
        # penalising actions if model cannot be created
        result = -1

    return result


def compute_objective(cfg, history):
    if '__' in cfg or cfg in history:
        split_cfg = cfg.split('__')
        kind = split_cfg[1] if len(split_cfg) > 1 else 'last'
        mname = split_cfg[0]
        if kind == 'min':
            return min(history[mname])
        elif kind == 'max':
            return max(history[mname])
        else: # 'last' or else
            return history[mname][-1]
    else:
        func = util.load_attr_from(cfg)
        return func(history)
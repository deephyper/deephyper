from random import random

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
    (t_X, t_y), (v_X, v_y) = load_data() if kwargs is None else load_data(**kwargs)
    print('[PARAM] Data loaded')

    # Set data shape
    input_shape = list(np.shape(t_X))[1:]
    output_shape = list(np.shape(t_y))[1:]

    config['data'] = {
        'train_X': t_X,
        'train_Y': t_y,
        'valid_X': v_X,
        'valid_Y': v_y
    }

    structure = config['create_structure']['func'](input_shape, output_shape, **config['create_structure']['kwargs'])
    arch_seq = config['arch_seq']
    structure.set_ops(arch_seq)

    if config['regression']:
        if config.get('preprocessing') is not None:
            preprocessing = util.load_attr_from(config['preprocessing']['func'])
            config['preprocessing']['func'] = preprocessing
        else:
            config['preprocessing'] = None

        model = structure.create_model()
        trainer = TrainerRegressorTrainValid(config=config, model=model)
    else:
        model = structure.create_model(activation='softmax')
        trainer = TrainerClassifierTrainValid(config=config, model=model)

    result = -trainer.train() if config['regression'] else trainer.train()
    return result

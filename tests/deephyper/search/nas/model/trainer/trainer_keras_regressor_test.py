import json
import sys
from random import random

import numpy as np
import pytest
from tensorflow.keras.utils import plot_model

from deephyper.search import util
from deephyper.search.nas.model.trainer.train_valid import \
    TrainerTrainValid


@pytest.mark.slow
def test_trainer_regressor_train_valid_with_one_input():
    from deephyper.benchmark.nas.linearReg.problem import Problem
    config = Problem.space

    config['hyperparameters']['num_epochs'] = 2

    # load functions
    load_data = util.load_attr_from(config['load_data']['func'])
    config['load_data']['func'] = load_data
    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    # Loading data
    kwargs = config['load_data'].get('kwargs')
    (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

    print('[PARAM] Data loaded')
    # Set data shape
    input_shape = np.shape(tX)[1:]  # interested in shape of data not in length
    output_shape = np.shape(ty)[1:]

    config['data'] = {
        'train_X': tX,
        'train_Y': ty,
        'valid_X': vX,
        'valid_Y': vy
    }

    structure = config['create_structure']['func'](
        input_shape, output_shape, **config['create_structure']['kwargs'])
    arch_seq = [random() for i in range(structure.num_nodes)]
    print('arch_seq: ', arch_seq)
    structure.set_ops(arch_seq)
    structure.draw_graphviz('trainer_keras_regressor_test.dot')

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
    else:
        config['preprocessing'] = None

    model = structure.create_model()
    plot_model(model, to_file='trainer_keras_regressor_test.png',
               show_shapes=True)

    trainer = TrainerTrainValid(config=config, model=model)

    res = trainer.train()
    assert res != sys.float_info.max


@pytest.mark.slow
def test_trainer_regressor_train_valid_with_multiple_ndarray_inputs():
    from deephyper.benchmark.nas.linearRegMultiInputs.problem import Problem
    config = Problem.space

    config['hyperparameters']['num_epochs'] = 2

    # load functions
    load_data = util.load_attr_from(config['load_data']['func'])
    config['load_data']['func'] = load_data
    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    # Loading data
    kwargs = config['load_data'].get('kwargs')
    (tX, ty), (vX, vy) = load_data() if kwargs is None else load_data(**kwargs)

    print('[PARAM] Data loaded')
    # Set data shape
    # interested in shape of data not in length
    input_shape = [np.shape(itX)[1:] for itX in tX]
    output_shape = list(np.shape(ty))[1:]

    config['data'] = {
        'train_X': tX,
        'train_Y': ty,
        'valid_X': vX,
        'valid_Y': vy
    }

    structure = config['create_structure']['func'](
        input_shape, output_shape, **config['create_structure']['kwargs'])
    arch_seq = [random() for i in range(structure.num_nodes)]
    print('arch_seq: ', arch_seq)
    structure.set_ops(arch_seq)
    structure.draw_graphviz('trainer_keras_regressor_test.dot')

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
    else:
        config['preprocessing'] = None

    model = structure.create_model()
    plot_model(model, to_file='trainer_keras_regressor_test.png',
               show_shapes=True)

    trainer = TrainerTrainValid(config=config, model=model)

    res = trainer.train()
    assert res != sys.float_info.max


@pytest.mark.slow
def test_trainer_regressor_train_valid_with_multiple_generator_inputs():
    from deephyper.benchmark.nas.linearRegMultiInputsGen.problem import Problem
    config = Problem.space

    config['hyperparameters']['num_epochs'] = 2

    # load functions
    load_data = util.load_attr_from(config['load_data']['func'])
    config['load_data']['func'] = load_data
    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    # Loading data
    kwargs = config['load_data'].get('kwargs')
    # (t_X, t_y), (v_X, v_y) = load_data() if kwargs is None else load_data(**kwargs)
    data = load_data() if kwargs is None else load_data(**kwargs)
    print('[PARAM] Data loaded')

    # Set data shape
    config['data'] = data
    input_shape = [data['shapes'][0][f'input_{i}']
                   for i in range(len(data['shapes'][0]))]
    output_shape = data['shapes'][1]

    structure = config['create_structure']['func'](
        input_shape, output_shape, **config['create_structure']['kwargs'])
    arch_seq = [random() for i in range(structure.num_nodes)]
    print('arch_seq: ', arch_seq)
    structure.set_ops(arch_seq)
    structure.draw_graphviz('trainer_keras_regressor_test.dot')

    if config.get('preprocessing') is not None:
        preprocessing = util.load_attr_from(config['preprocessing']['func'])
        config['preprocessing']['func'] = preprocessing
    else:
        config['preprocessing'] = None

    model = structure.create_model()
    plot_model(model, to_file='trainer_keras_regressor_test.png',
               show_shapes=True)

    trainer = TrainerTrainValid(config=config, model=model)

    res = trainer.train()
    assert res != sys.float_info.max


if __name__ == '__main__':
    # test_trainer_regressor_train_valid_with_one_input()
    # test_trainer_regressor_train_valid_with_multiple_ndarray_inputs()
    test_trainer_regressor_train_valid_with_multiple_generator_inputs()

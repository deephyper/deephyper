import numpy as np
from deephyper.search import util
from deephyper.core.exceptions.problem import WrongProblemObjective

logger = util.conf_logger('deephyper.search.nas.run')

def load_config(config):
    # ! load functions
    config['load_data']['func'] = util.load_attr_from(config['load_data']['func'])

    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    if not config.get('preprocessing') is None:
        config['preprocessing']['func'] = util.load_attr_from(config['preprocessing']['func'])
    else:
        config['preprocessing'] = None

    if '.' in config['objective']:
        config['objective'] = util.load_attr_from(config['objective'])


def setup_data(config):
    # Loading data
    load_data = config['load_data']['func']
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

    return input_shape, output_shape


def setup_structure(config, input_shape, output_shape):

    create_structure = config['create_structure']['func']
    cs_kwargs = config['create_structure'].get('kwargs')
    if cs_kwargs is None:
        structure = create_structure(input_shape, output_shape)
    else:
        structure = create_structure(input_shape, output_shape, **cs_kwargs)

    arch_seq = config['arch_seq']
    logger.info(f'actions list: {arch_seq}')
    structure.set_ops(arch_seq)

    return structure


def compute_objective(objective, history):
    if type(objective) is str \
        and ('__' in objective or objective in history):

        split_objective = objective.split('__')
        kind = split_objective[1] if len(split_objective) > 1 else 'last'
        mname = split_objective[0]
        if kind == 'min':
            return min(history[mname])
        elif kind == 'max':
            return max(history[mname])
        else: # 'last' or else
            return history[mname][-1]
    elif callable(objective):
        func = objective
        return func(history)
    else:
        raise WrongProblemObjective(objective)


def preproc_trainer(config):

    if type(config['objective']) is str:
        last_only = '__last' in config['objective']
    else: # should be callable
        last_only = '__last' in config['objective'].__name__

    with_pred = not type(config['objective']) is str \
        and 'with_pred' in config['objective'].__name__
    return last_only, with_pred

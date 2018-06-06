import argparse
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from deephyper.benchmarks import util

def create_parser():
    'command line parser for keras'

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 0.1')
    parser.add_argument('--backend', action='store',
                        dest='backend',
                        nargs='?', const=1, type=str, default='tensorflow',
                        choices=['tensorflow', 'theano', 'cntk'],
                        help='Keras backend')
    parser.add_argument('--activation', action='store',
                        dest='activation',
                        nargs='?', const=1, type=str, default='relu',
                        choices=['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
                                 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU'],
                        help='type of activation function hidden layer')
    parser.add_argument('--loss', action='store', dest='loss',
                        nargs='?', const=1, type=str, default='mae',
                        choices=['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'categorical_hinge', 'hinge', 'logcosh',
                                 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
                                 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'],
                        help='type of loss')
    parser.add_argument('--epochs', action='store', dest='epochs',
                        nargs='?', const=2, type=int, default='2',
                        help='number of epochs')
    parser.add_argument('--batch_size', action='store', dest='batch_size',
                        nargs='?', const=1, type=int, default='128',
                        help='batch size')
    parser.add_argument('--init', action='store', dest='init',
                        nargs='?', const=1, type=str, default='normal',
                        choices=['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                                 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal',
                                 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform'],
                        help='type of initialization')
    parser.add_argument('--dropout', action='store', dest='dropout', nargs='?', const=1, type=float, default=0.0,
                        help=' float [0, 1). Fraction of the input units to drop')

    parser.add_argument('--optimizer', action='store',
                        dest='optimizer',
                        nargs='?', const=1, type=str, default='sgd',
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
                        help='type of optimizer')

    parser.add_argument('--patience', action='store', dest='patience',
                        nargs='?', const=2, type=int, default='10',
                        help='number of patience epochs for early stopping')

    parser.add_argument('--delta', action='store', dest='delta',
                        nargs='?', const=1, type=float, default=0.0001,
                        help='float >= 0. min delta for early stopping')

    # common optimizer parameters
    parser.add_argument('--clipnorm', action='store', dest='clipnorm',
                        nargs='?', const=1, type=float, default=1.0,
                        help='float >= 0. Gradients will be clipped when their \
                        L2 norm exceeds this value.')
    parser.add_argument('--clipvalue', action='store', dest='clipvalue',
                        nargs='?', const=1, type=float, default=0.5,
                        help='float >= 0. Gradients will be clipped when their \
                        absolute value exceeds this value.')

    # optimizer parameters
    parser.add_argument('--learning_rate', action='store', dest='lr',
                        nargs='?', const=1, type=float, default=0.01,
                        help='float >= 0. Learning rate')
    parser.add_argument('--momentum', action='store', dest='momentum',
                        nargs='?', const=1, type=float, default=0.0,
                        help='float >= 0. Parameter updates momentum')
    parser.add_argument('--decay', action='store', dest='decay',
                        nargs='?', const=1, type=float, default=0.0,
                        help='float >= 0. Learning rate decay over each update')
    parser.add_argument('--nesterov', action='store', dest='nesterov',
                        nargs='?', const=1, type=util.str2bool, default=False,
                        help='boolean. Whether to apply Nesterov momentum?')
    parser.add_argument('--rho', action='store', dest='rho',
                        nargs='?', const=1, type=float, default=0.9,
                        help='float >= 0')
    parser.add_argument('--epsilon', action='store',
                        dest='epsilon',
                        nargs='?', const=1, type=float, default=1e-08,
                        help='float >= 0')
    parser.add_argument('--beta1', action='store', dest='beta1',
                        nargs='?', const=1, type=float, default=0.9,
                        help='float >= 0')
    parser.add_argument('--beta2', action='store', dest='beta2',
                        nargs='?', const=1, type=float, default=0.999,
                        help='float >= 0')

    # model and data I/O options
    parser.add_argument('--model_path', help="path from which models are loaded/saved", default='')
    parser.add_argument('--data_source', help="location of dataset to load", default='')
    parser.add_argument('--stage_in_destination', help="if provided; cache data at this location", 
                        default='')

    parser.add_argument('--timeout', help="benchmark timeout in mins", 
                        action='store', dest='timeout',
                        nargs='?', const=1, type=float, 
                        default=60.0)
    return(parser)

def return_optimizer(param_dict):
    optimizer = None
    if param_dict['optimizer'] == 'sgd':
        optimizer = SGD(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'rmsprop':
        optimizer = RMSprop(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'adagrad':
        optimizer = Adagrad(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'adadelta':
        optimizer = Adadelta(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'adam':
        optimizer = Adam(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'adamax':
        optimizer = Adamax(lr=param_dict['lr'])
    elif param_dict['optimizer'] == 'nadam':
        optimizer = Nadam(lr=param_dict['lr'])
                        
    return(optimizer)

def fill_missing_defaults(augment_parser_fxn, param_dict):
    '''Build an augmented parser; return param_dict filled in
    with missing values that were not supplied directly'''
    def_parser = create_parser()
    def_parser = augment_parser_fxn(def_parser)
    default_params = vars(def_parser.parse_args(''))

    missing = (k for k in default_params if k not in param_dict)
    for k in missing:
        param_dict[k] = default_params[k]
    return param_dict

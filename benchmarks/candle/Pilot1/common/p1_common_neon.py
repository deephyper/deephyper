from __future__ import absolute_import

from neon import transforms
from neon import optimizers
from neon import initializers

from neon.util.argparser import NeonArgparser
import argparse


def add_nonstandard_to_neon_parser(parser):
    """Parse command-line arguments that are not standard in the neon argparse. Ignore if not present.
        
        Parameters
        ----------
        parser : neon argparse
            parser for command-line options
    """
    
    # Model definition
    # Model Architecture
    parser.add_argument("--dense", action="store", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="number of units in fully connected layers in an integer array")

    return parser


def get_function(name):
    mapping = {}
    
    # activation
    mapping['relu'] = transforms.activation.Rectlin
    mapping['sigmoid'] = transforms.activation.Logistic
    mapping['tanh'] = transforms.activation.Tanh
    mapping['linear'] = transforms.activation.Identity
    
    # loss
    mapping['mse'] = transforms.cost.MeanSquared
    mapping['binary_crossentropy'] = transforms.cost.CrossEntropyBinary
    mapping['categorical_crossentropy'] = transforms.cost.CrossEntropyMulti
    mapping['smoothL1'] = transforms.SmoothL1Loss
    
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No neon function found for "{}"'.format(name))
    
    return mapped



def build_optimizer(type, lr, kerasDefaults):
    
    schedule = optimizers.optimizer.Schedule() # constant lr (equivalent to default keras setting)

    if type == 'sgd':
        return optimizers.GradientDescentMomentum(learning_rate=lr,
                                                  momentum_coef=kerasDefaults['momentum_sgd'],
                                                  nesterov=kerasDefaults['nesterov_sgd'],
                                                  #gradient_clip_norm=kerasDefaults['clipnorm'],
                                                  #gradient_clip_value=kerasDefaults['clipvalue'],
                                                  schedule=schedule)
    
    elif type == 'rmsprop':
        return optimizers.RMSProp(learning_rate=lr,
                                  decay_rate=kerasDefaults['rho'],
                                  epsilon=kerasDefaults['epsilon'],
                                  #gradient_clip_norm=kerasDefaults['clipnorm'],
                                  #gradient_clip_value=kerasDefaults['clipvalue'],
                                  schedule=schedule)

    elif type == 'adagrad':
        return optimizers.Adagrad(learning_rate=lr,
                                  epsilon=kerasDefaults['epsilon'])#,
                                  #gradient_clip_norm=kerasDefaults['clipnorm'],
                                  #gradient_clip_value=kerasDefaults['clipvalue'])

    elif type == 'adadelta':
        return optimizers.Adadelta(epsilon=kerasDefaults['epsilon'],
                                   decay=kerasDefaults['rho'])#,
                                   #gradient_clip_norm=kerasDefaults['clipnorm'],
                                   #gradient_clip_value=kerasDefaults['clipvalue'])

    elif type == 'adam':
        return optimizers.Adam(learning_rate=lr, beta_1=kerasDefaults['beta_1'],
                               beta_2=kerasDefaults['beta_2'],
                               epsilon=kerasDefaults['epsilon'])#,
                               #gradient_clip_norm=kerasDefaults['clipnorm'],
                               #gradient_clip_value=kerasDefaults['clipvalue'])



def build_initializer(type, kerasDefaults, constant=0.):
    
    if type == 'constant':
        return initializers.Constant(val=constant)
    
    elif type == 'uniform':
        return initializers.Uniform(low=kerasDefaults['minval_uniform'],
                                  high=kerasDefaults['maxval_uniform'])

    elif type == 'normal':
        return initializers.Gaussian(loc=kerasDefaults['mean_normal'],
                                  scale=kerasDefaults['stddev_normal'])

    elif type == 'glorot_uniform':
        return initializers.GlorotUniform()

    elif type == 'lecun_uniform':
        return initializers.Xavier()

    elif type == 'he_normal':
        return initializers.Kaiming()


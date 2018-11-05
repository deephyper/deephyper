from __future__ import absolute_import

import torch 
import torch.nn.init
import torch.optim 
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
    
    # loss
    mapping['mse'] = torch.nn.MSELoss()
    #mapping['binary_crossentropy'] = transforms.cost.CrossEntropyBinary
    #mapping['categorical_crossentropy'] = transforms.cost.CrossEntropyMulti
    mapping['smoothL1'] = torch.nn.SmoothL1Loss()
    
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No pytorch function found for "{}"'.format(name))
    
    return mapped

def build_activation(type):

    # activation
     
    if type=='relu':
         return torch.nn.ReLU()
    elif type=='sigmoid':
         return torch.nn.Sigmoid()
    elif type=='tanh':
         return torch.nn.Tanh()
    #mapping['linear'] = transforms.activation.Identity
    


def build_optimizer(model, type, lr, kerasDefaults):
    
    #schedule = optimizers.optimizer.Schedule() # constant lr (equivalent to default keras setting)

    if type == 'sgd':
        return torch.optim.GradientDescentMomentum(model.parameters(), 
                                                  lr=lr,
                                                  momentum_coef=kerasDefaults['momentum_sgd'],
                                                  nesterov=kerasDefaults['nesterov_sgd'])
                                                  #schedule=schedule)
    
    elif type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), 
                                  lr=lr,
                                  alpha=kerasDefaults['rho'],
                                  eps=kerasDefaults['epsilon'])
                                  #schedule=schedule)

    elif type == 'adagrad':
        return torch.optim.Adagrad(model.parameters(),
                              lr=lr,
                              eps=kerasDefaults['epsilon'])

    elif type == 'adadelta':
        return torch.optim.Adadelta(model.parameters(),
                              eps=kerasDefaults['epsilon'],
                              rho=kerasDefaults['rho'])

    elif type == 'adam':
        return torch.optim.Adam(model.parameters(),
                               lr=lr, 
                               betas={kerasDefaults['beta_1'], kerasDefaults['beta_2']},
                               eps=kerasDefaults['epsilon'])



def build_initializer(weights, type, kerasDefaults, seed=None, constant=0.):
    
    if type == 'constant':
        return torch.nn.init.constant(weights,
                                    val=constant)
    
    elif type == 'uniform':
        return torch.nn.init.uniform(weights,
                                  a=kerasDefaults['minval_uniform'],
                                  b=kerasDefaults['maxval_uniform'])

    elif type == 'normal':
        return torch.nn.init.normal(weights,
                                  mean=kerasDefaults['mean_normal'],
                                  std=kerasDefaults['stddev_normal'])

    elif type == 'glorot_normal': # not quite Xavier
        return torch.nn.init.xavier_normal(weights)

    elif type == 'glorot_uniform':
        return torch.nn.init.xavier_uniform(weights)

    elif type == 'he_normal':
        return torch.nn.init.kaiming_uniform(weights)


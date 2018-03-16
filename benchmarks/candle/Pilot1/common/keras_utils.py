from __future__ import absolute_import

from keras import optimizers
from keras import initializers

def get_function(name):
    mapping = {}
    
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No keras function found for "{}"'.format(name))
    
    return mapped



def build_optimizer(type, lr, kerasDefaults):

    if type == 'sgd':
        return optimizers.SGD(lr=lr, decay=kerasDefaults['decay_lr'],
                              momentum=kerasDefaults['momentum_sgd'],
                              nesterov=kerasDefaults['nesterov_sgd'])#,
            #clipnorm=kerasDefaults['clipnorm'],
            #clipvalue=kerasDefaults['clipvalue'])
    
    elif type == 'rmsprop':
        return optimizers.RMSprop(lr=lr, rho=kerasDefaults['rho'],
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])#,
            #clipnorm=kerasDefaults['clipnorm'],
#clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adagrad':
        return optimizers.Adagrad(lr=lr,
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])#,
                                  #clipnorm=kerasDefaults['clipnorm'],
                                  #clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adadelta':
        return optimizers.Adadelta(lr=lr, rho=kerasDefaults['rho'],
                                   epsilon=kerasDefaults['epsilon'],
                                   decay=kerasDefaults['decay_lr'])#,
                                    #clipnorm=kerasDefaults['clipnorm'],
#clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adam':
        return optimizers.Adam(lr=lr, beta_1=kerasDefaults['beta_1'],
                               beta_2=kerasDefaults['beta_2'],
                               epsilon=kerasDefaults['epsilon'],
                               decay=kerasDefaults['decay_lr'])#,
                               #clipnorm=kerasDefaults['clipnorm'],
                               #clipvalue=kerasDefaults['clipvalue'])

# Not generally available
#    elif type == 'adamax':
#        return optimizers.Adamax(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               decay=kerasDefaults['decay_lr'])

#    elif type == 'nadam':
#        return optimizers.Nadam(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               schedule_decay=kerasDefaults['decay_schedule_lr'])



def build_initializer(type, kerasDefaults, seed=None, constant=0.):
    
    if type == 'constant':
        return initializers.Constant(value=constant)
    
    elif type == 'uniform':
        return initializers.RandomUniform(minval=kerasDefaults['minval_uniform'],
                                  maxval=kerasDefaults['maxval_uniform'],
                                  seed=seed)

    elif type == 'normal':
        return initializers.RandomNormal(mean=kerasDefaults['mean_normal'],
                                  stddev=kerasDefaults['stddev_normal'],
                                  seed=seed)

# Not generally available
#    elif type == 'glorot_normal':
#        return initializers.glorot_normal(seed=seed)

    elif type == 'glorot_uniform':
        return initializers.glorot_uniform(seed=seed)

    elif type == 'lecun_uniform':
        return initializers.lecun_uniform(seed=seed)

    elif type == 'he_normal':
        return initializers.he_normal(seed=seed)


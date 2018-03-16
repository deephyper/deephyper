from __future__ import absolute_import

import mxnet as mx
from mxnet import initializer
from mxnet import optimizer

def plot_network(net, filename):
    try:
        dot = mx.viz.plot_network(net)
    except ImportError:
        return
    try:
        dot.render(filename, view=False)
        print('Plotted network architecture in {}'.format(filename+'.pdf'))
    except Exception:
        return


def get_function(name):
    mapping = {}
    
    # activation
    #mapping['linear'] = transforms.activation.Identity
    
    # loss
    mapping['mse'] = mx.metric.MSE
    #mapping['binary_crossentropy'] = transforms.cost.CrossEntropyBinary
    mapping['categorical_crossentropy'] = mx.metric.CrossEntropy
    #mapping['smoothL1'] = transforms.SmoothL1Loss
    
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mxnet function found for "{}"'.format(name))
    
    return mapped


def build_optimizer(type, lr, kerasDefaults):


    if type == 'sgd':
        if kerasDefaults['nesterov_sgd']:
            return optimizer.NAG(learning_rate=lr,
                                 momentum=kerasDefaults['momentum_sgd'],
                                 #rescale_grad=kerasDefaults['clipnorm'],
                                 #clip_gradient=kerasDefaults['clipvalue'],
                                 lr_scheduler=None)
        else:
            return optimizer.SGD(learning_rate=lr,
                                 momentum=kerasDefaults['momentum_sgd'],
                                 #rescale_grad=kerasDefaults['clipnorm'],
                                 #clip_gradient=kerasDefaults['clipvalue'],
                                 lr_scheduler=None)
    
    elif type == 'rmsprop':
        return optimizer.RMSProp(learning_rate=lr,
                                 gamma1=kerasDefaults['rho'],
                                 epsilon=kerasDefaults['epsilon'],
                                 centered=False,
                                 #rescale_grad=kerasDefaults['clipnorm'],
                                 #clip_gradient=kerasDefaults['clipvalue'],
                                 lr_scheduler=None)

    elif type == 'adagrad':
        return optimizer.AdaGrad(learning_rate=lr,
                                 epsilon=kerasDefaults['epsilon'])#,
                                 #rescale_grad=kerasDefaults['clipnorm'],
                                 #clip_gradient=kerasDefaults['clipvalue'])

    elif type == 'adadelta':
        return optimizer.AdaDelta(epsilon=kerasDefaults['epsilon'],
                                  rho=kerasDefaults['rho'])#,
                                  #rescale_grad=kerasDefaults['clipnorm'],
                                  #clip_gradient=kerasDefaults['clipvalue'])

    elif type == 'adam':
        return optimizer.Adam(learning_rate=lr, beta_1=kerasDefaults['beta_1'],
                              beta_2=kerasDefaults['beta_2'],
                              epsilon=kerasDefaults['epsilon'])#,
                              #rescale_grad=kerasDefaults['clipnorm'],
                              #clip_gradient=kerasDefaults['clipvalue'])


def build_initializer(type, kerasDefaults, constant=0.):
    
    if type == 'constant':
        return initializer.Constant(constant)
    
    elif type == 'uniform':
        return initializer.Uniform(scale=kerasDefaults['maxval_uniform'])

    elif type == 'normal':
        return initializer.Normal(sigma=kerasDefaults['stddev_normal'])

    elif type == 'glorot_uniform':
        return initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3.)

    elif type == 'lecun_uniform':
        return initializers.Xavier(rnd_type='uniform', factor_type='in', magnitude=3.)

    elif type == 'he_normal':
        return initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2.)


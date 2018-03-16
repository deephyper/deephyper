from __future__ import division, print_function

import argparse
import logging

import numpy as np

import mxnet as mx
from mxnet.io import DataBatch, DataIter

# # For non-interactive plotting
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import p1b3
import p1_common
import p1_common_mxnet


def get_p1b3_parser():
    
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Drug Response Regressor - Pilot 1 Benchmark 3')
        
    return p1b3.common_parser(parser)


class ConcatDataIter(DataIter):
    """Data iterator for concatenated features
    """

    def __init__(self, data_loader,
                 partition='train',
                 batch_size=32,
                 num_data=None,
                 shape=None):
        super(ConcatDataIter, self).__init__()
        self.data = data_loader
        self.batch_size = batch_size
        self.gen = p1b3.DataGenerator(data_loader, partition=partition, batch_size=batch_size, shape=shape, concat=True)
        self.num_data = num_data or self.gen.num_data
        self.cursor = 0
        self.gen = self.gen.flow()

    @property
    def provide_data(self):
        return [('concat_features', (self.batch_size, self.data.input_dim))]

    @property
    def provide_label(self):
        return [('growth', (self.batch_size,))]

    def reset(self):
        self.cursor = 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor <= self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            x, y = next(self.gen)
            return DataBatch(data=[mx.nd.array(x)], label=[mx.nd.array(y)])
        else:
            raise StopIteration


def main():
    # Get command-line parameters
    parser = get_p1b3_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b3.read_config_file(args.config_file)
    #print ('Params:', fileParameters)

    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)

    # Determine verbosity level
    loggingLevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loggingLevel, format='')
    # Construct extension to save model
    ext = p1b3.extension_from_parameters(gParameters, '.neon')

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Build dataset loader object
    loader = p1b3.DataLoader(seed=seed, dtype=gParameters['datatype'],
                             val_split=gParameters['validation_split'],
                             test_cell_split=gParameters['test_cell_split'],
                             cell_features=gParameters['cell_features'],
                             drug_features=gParameters['drug_features'],
                             feature_subsample=gParameters['feature_subsample'],
                             scaling=gParameters['scaling'],
                             scramble=gParameters['scramble'],
                             min_logconc=gParameters['min_logconc'],
                             max_logconc=gParameters['max_logconc'],
                             subsample=gParameters['subsample'],
                             category_cutoffs=gParameters['category_cutoffs'])
    

    net = mx.sym.Variable('concat_features')
    out = mx.sym.Variable('growth')
    
    # Initialize weights and learning rule
    initializer_weights = p1_common_mxnet.build_initializer(gParameters['initialization'], kerasDefaults)
    initializer_bias = p1_common_mxnet.build_initializer('constant', kerasDefaults, 0.)
    init = mx.initializer.Mixed(['bias', '.*'], [initializer_bias, initializer_weights])
    
    activation = gParameters['activation']

    # Define model architecture
    layers = []
    reshape = None

    if 'dense' in gParameters: # Build dense layers
        for layer in gParameters['dense']:
            if layer:
                net = mx.sym.FullyConnected(data=net, num_hidden=layer)
                net = mx.sym.Activation(data=net, act_type=activation)
            if gParameters['drop']:
                net = mx.sym.Dropout(data=net, p=gParameters['drop'])
    else: # Build convolutional layers
        net = mx.sym.Reshape(data=net, shape=(gParameters['batch_size'], 1, loader.input_dim, 1))
        layer_list = list(range(0, len(args.convolution), 3))
        for l, i in enumerate(layer_list):
            nb_filter = gParameters['conv'][i]
            filter_len = gParameters['conv'][i+1]
            stride = gParameters['conv'][i+2]
            if nb_filter <= 0 or filter_len <= 0 or stride <= 0:
                break
            net = mx.sym.Convolution(data=net, num_filter=nb_filter, kernel=(filter_len, 1), stride=(stride, 1))
            net = mx.sym.Activation(data=net, act_type=activation)
            if gParameters['pool']:
                net = mx.sym.Pooling(data=net, pool_type="max", kernel=(gParameters['pool'], 1), stride=(1, 1))
        net = mx.sym.Flatten(data=net)
        
        
        reshape = (1, loader.input_dim, 1)
        layer_list = list(range(0, len(gParameters['conv']), 3))
        for l, i in enumerate(layer_list):
            nb_filter = gParameters['conv'][i]
            filter_len = gParameters['conv'][i+1]
            stride = gParameters['conv'][i+2]
            # print(nb_filter, filter_len, stride)
            # fshape: (height, width, num_filters).
            layers.append(Conv((1, filter_len, nb_filter), strides={'str_h':1, 'str_w':stride},
                init=initializer_weights, activation=activation))
            if gParameters['pool']:
                layers.append(Pooling((1, gParameters['pool'])))

    net = mx.sym.FullyConnected(data=net, num_hidden=1)
    net = mx.symbol.LinearRegressionOutput(data=net, label=out)

    # Display model
    p1_common_mxnet.plot_network(net, 'net'+ext)

    # Define mxnet data iterators
    train_samples = int(loader.n_train)
    val_samples = int(loader.n_val)

    if 'train_samples' in gParameters:
        train_samples = gParameters['train_samples']
    if 'val_samples' in gParameters:
        val_samples = gParameters['val_samples']

    train_iter = ConcatDataIter(loader, batch_size=gParameters['batch_size'], num_data=train_samples)
    val_iter = ConcatDataIter(loader, partition='val', batch_size=gParameters['batch_size'], num_data=val_samples)

    devices = mx.cpu()
    if gParameters['gpus']:
        devices = [mx.gpu(i) for i in gParameters['gpus']]

    mod = mx.mod.Module(net,
                        data_names=('concat_features',),
                        label_names=('growth',),
                        context=devices)


    # Define optimizer
    optimizer = p1_common_mxnet.build_optimizer(gParameters['optimizer'],
                                                gParameters['learning_rate'],
                                                kerasDefaults)

    # Seed random generator for training
    mx.random.seed(seed)

    freq_log = 1

#initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    mod.fit(train_iter, eval_data=val_iter,
            eval_metric=gParameters['loss'],
            optimizer=optimizer,
            num_epoch=gParameters['epochs'],
            initializer=init,
            epoch_end_callback = mx.callback.Speedometer(gParameters['batch_size'], 20))


if __name__ == '__main__':
    main()

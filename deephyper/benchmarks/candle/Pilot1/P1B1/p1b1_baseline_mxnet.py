from __future__ import print_function

import numpy as np

import argparse

import mxnet as mx
from mxnet.io import DataBatch, DataIter


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b1
import p1_common
import p1_common_mxnet


def get_p1b1_parser():

	parser = argparse.ArgumentParser(prog='p1b1_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Autoencoder - Pilot 1 Benchmark 1')

	return p1b1.common_parser(parser)



def main():

    # Get command-line parameters
    parser = get_p1b1_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b1.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)

    # Construct extension to save model
    ext = p1b1.extension_from_parameters(gParameters, '.mx')
    logfile = args.logfile if args.logfile else args.save+ext+'.log'
    p1b1.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Load dataset
    X_train, X_val, X_test = p1b1.load_data(gParameters, seed)
    
    print ("Shape X_train: ", X_train.shape)
    print ("Shape X_val: ", X_val.shape)
    print ("Shape X_test: ", X_test.shape)

    print ("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    print ("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print ("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))


    # Set input and target to X_train
    train_iter = mx.io.NDArrayIter(X_train, X_train, gParameters['batch_size'], shuffle=gParameters['shuffle'])
    val_iter = mx.io.NDArrayIter(X_val, X_val, gParameters['batch_size'])
    test_iter = mx.io.NDArrayIter(X_test, X_test, gParameters['batch_size'])
    
    net = mx.sym.Variable('data')
    out = mx.sym.Variable('softmax_label')
    input_dim = X_train.shape[1]
    output_dim = input_dim

    # Initialize weights and learning rule
    initializer_weights = p1_common_mxnet.build_initializer(gParameters['initialization'], kerasDefaults)
    initializer_bias = p1_common_mxnet.build_initializer('constant', kerasDefaults, 0.)
    init = mx.initializer.Mixed(['bias', '.*'], [initializer_bias, initializer_weights])
    
    activation = gParameters['activation']

    # Define Autoencoder architecture
    layers = gParameters['dense']
    
    if layers != None:
        if type(layers) != list:
            layers = list(layers)
        # Encoder Part
        for i,l in enumerate(layers):
            net = mx.sym.FullyConnected(data=net, num_hidden=l)
            net = mx.sym.Activation(data=net, act_type=activation)
        # Decoder Part
        for i,l in reversed( list(enumerate(layers)) ):
            if i < len(layers)-1:
                net = mx.sym.FullyConnected(data=net, num_hidden=l)
                net = mx.sym.Activation(data=net, act_type=activation)
                    
    net = mx.sym.FullyConnected(data=net, num_hidden=output_dim)
    #net = mx.sym.Activation(data=net, act_type=activation)
    net = mx.symbol.LinearRegressionOutput(data=net, label=out)


    # Display model
    p1_common_mxnet.plot_network(net, 'net'+ext)

    # Define context
    devices = mx.cpu()
    if gParameters['gpus']:
        devices = [mx.gpu(i) for i in gParameters['gpus']]
    

    # Build Autoencoder model
    ae = mx.mod.Module(symbol=net, context=devices)

    # Define optimizer
    optimizer = p1_common_mxnet.build_optimizer(gParameters['optimizer'],
                                                gParameters['learning_rate'],
                                                kerasDefaults)

    # Seed random generator for training
    mx.random.seed(seed)

    freq_log = 1
    ae.fit(train_iter, eval_data=val_iter,
           eval_metric=gParameters['loss'],
           optimizer=optimizer,
           num_epoch=gParameters['epochs'])#,
           #epoch_end_callback = mx.callback.Speedometer(gParameters['batch_size'], freq_log))

    # model save
    #save_filepath = "model_ae_" + ext
    #ae.save(save_filepath)

    # Evalute model on test set
    X_pred = ae.predict(test_iter).asnumpy()
    #print ("Shape X_pred: ", X_pred.shape)
    
    scores = p1b1.evaluate_autoencoder(X_pred, X_test)
    print('Evaluation on test data:', scores)

    diff = X_pred - X_test
    plt.hist(diff.ravel(), bins='auto')
    plt.title("Histogram of Errors with 'auto' bins")
    plt.savefig('histogram_mx.png')


if __name__ == '__main__':
    main()

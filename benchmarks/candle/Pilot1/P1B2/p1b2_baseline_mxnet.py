from __future__ import print_function

import argparse

import numpy as np

import mxnet as mx
from mxnet.io import DataBatch, DataIter

import p1b2
import p1_common
import p1_common_mxnet



def get_p1b2_parser():

	parser = argparse.ArgumentParser(prog='p1b2_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train Classifier - Pilot 1 Benchmark 2')

	return p1b2.common_parser(parser)


def main():
    
    # Get command-line parameters
    parser = get_p1b2_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b2.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)

    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, '.mx')
    logfile = args.logfile if args.logfile else args.save+ext+'.log'
    p1b2.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']
    
    # Load dataset
    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data(gParameters, seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data_one_hot(gParameters, seed)

    print ("Shape X_train: ", X_train.shape)
    print ("Shape X_val: ", X_val.shape)
    print ("Shape X_test: ", X_test.shape)
    print ("Shape y_train: ", y_train.shape)
    print ("Shape y_val: ", y_val.shape)
    print ("Shape y_test: ", y_test.shape)

    print ("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    print ("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print ("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))
    print ("Range y_train --> Min: ", np.min(y_train), ", max: ", np.max(y_train))
    print ("Range y_val --> Min: ", np.min(y_val), ", max: ", np.max(y_val))
    print ("Range y_test --> Min: ", np.min(y_test), ", max: ", np.max(y_test))


    # Set input and target to X_train
    train_iter = mx.io.NDArrayIter(X_train, y_train, gParameters['batch_size'], shuffle=gParameters['shuffle'])
    val_iter = mx.io.NDArrayIter(X_val, y_val, gParameters['batch_size'])
    test_iter = mx.io.NDArrayIter(X_test, y_test, gParameters['batch_size'])
    
    net = mx.sym.Variable('data')#X')
    out = mx.sym.Variable('softmax_label')#y')
    num_classes = y_train.shape[1]

    # Initialize weights and learning rule
    initializer_weights = p1_common_mxnet.build_initializer(gParameters['initialization'], kerasDefaults)
    initializer_bias = p1_common_mxnet.build_initializer('constant', kerasDefaults, 0.)
    init = mx.initializer.Mixed(['bias', '.*'], [initializer_bias, initializer_weights])
    
    activation = gParameters['activation']
    
    # Define MLP architecture
    layers = gParameters['dense']
    
    if layers != None:
        if type(layers) != list:
            layers = list(layers)
        for i,l in enumerate(layers):
            net = mx.sym.FullyConnected(data=net, num_hidden=l)
            net = mx.sym.Activation(data=net, act_type=activation)
            if gParameters['drop']:
                net = mx.sym.Dropout(data=net, p=gParameters['drop'])

    net = mx.sym.FullyConnected(data=net, num_hidden=num_classes)# 1)
    net = mx.symbol.SoftmaxOutput(data=net, label=out)

    # Display model
    p1_common_mxnet.plot_network(net, 'net'+ext)

    devices = mx.cpu()
    if gParameters['gpus']:
        devices = [mx.gpu(i) for i in gParameters['gpus']]
                          
    # Build MLP model
    mlp = mx.mod.Module(symbol=net,
                        context=devices)

    # Define optimizer
    optimizer = p1_common_mxnet.build_optimizer(gParameters['optimizer'],
                                                gParameters['learning_rate'],
                                                kerasDefaults)
                                                
    metric = p1_common_mxnet.get_function(gParameters['loss'])()

    # Seed random generator for training
    mx.random.seed(seed)

    mlp.fit(train_iter, eval_data=val_iter,
#            eval_metric=metric,
            optimizer=optimizer,
            num_epoch=gParameters['epochs'],
            initializer=init
            )

    # model save
    #save_filepath = "model_mlp_" + ext
    #mlp.save(save_filepath)

    # Evalute model on test set
    y_pred = mlp.predict(test_iter).asnumpy()
    #print ("Shape y_pred: ", y_pred.shape)
    scores = p1b2.evaluate_accuracy_one_hot(y_pred, y_test)
    print('Evaluation on test data:', scores)


if __name__ == '__main__':
    main()

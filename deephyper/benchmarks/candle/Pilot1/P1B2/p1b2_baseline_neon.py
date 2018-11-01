from __future__ import division, print_function

import os
import sys
import argparse
import logging

import numpy as np

import neon
from neon.util.argparser import NeonArgparser
from neon.data import ArrayIterator
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.layers import GeneralizedCost, Affine, Dropout, Reshape
from neon.models import Model
from neon.backends import gen_backend
from neon.transforms import Accuracy

#from neon import logger as neon_logger


import p1b2
import p1_common
import p1_common_neon


def get_p1b2_parser():
    
    # Construct neon arg parser. It generates a large set of options by default
    parser = NeonArgparser(__doc__)
    # Specify the default config_file
    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(p1b2.file_path, 'p1b2_default_model.txt'),
                        help="specify model configuration file")
    
    # Parse other options that are not included on neon arg parser
    parser = p1_common.get_p1_common_parser(parser)
    
    
    return parser


def main():
    # Get command-line parameters
    parser = get_p1b2_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b2.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    
    # Correct for arguments set by default by neon parser
    # (i.e. instead of taking the neon parser default value fall back to the config file,
    # if effectively the command-line was used, then use the command-line value)
    # This applies to conflictive parameters: batch_size, epochs and rng_seed
    if not any("--batch_size" in ag or "-z" in ag for ag in sys.argv):
        args.batch_size = fileParameters['batch_size']
    if not any("--epochs" in ag or "-e" in ag for ag in sys.argv):
        args.epochs = fileParameters['epochs']
    if not any("--rng_seed" in ag or "-r" in ag for ag in sys.argv):
        args.rng_seed = fileParameters['rng_seed']

    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)
    
    # Determine verbosity level
    loggingLevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loggingLevel, format='')
    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, '.neon')

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Load dataset
    #(X_train, y_train), (X_test, y_test) = p1b2.load_data(gParameters, seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data(gParameters, seed)

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

    input_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    output_dim = num_classes # The backend will represent the classes using one-hot representation (but requires an integer class as input !)
    
    # Re-generate the backend after consolidating parsing and file config
    gen_backend(backend=args.backend,
                rng_seed=seed,
                device_id=args.device_id,
                batch_size=gParameters['batch_size'],
                datatype=gParameters['datatype'],
                max_devices=args.max_devices,
                compat_mode=args.compat_mode)

    train = ArrayIterator(X=X_train, y=y_train, nclass=num_classes)
    val = ArrayIterator(X=X_val, y=y_val, nclass=num_classes)
    test = ArrayIterator(X=X_test, y=y_test, nclass=num_classes)

    # Initialize weights and learning rule
    initializer_weights = p1_common_neon.build_initializer(gParameters['initialization'], kerasDefaults, seed)
    initializer_bias = p1_common_neon.build_initializer('constant', kerasDefaults, 0.)

    activation = p1_common_neon.get_function(gParameters['activation'])()


    # Define MLP architecture
    layers = []
    reshape = None

    for layer in gParameters['dense']:
        if layer:
            layers.append(Affine(nout=layer, init=initializer_weights, bias=initializer_bias, activation=activation))
        if gParameters['drop']:
            layers.append(Dropout(keep=(1-gParameters['drop'])))

    layers.append(Affine(nout=output_dim, init=initializer_weights, bias=initializer_bias, activation=activation))

    # Build MLP model
    mlp = Model(layers=layers)

    # Define cost and optimizer
    cost = GeneralizedCost(p1_common_neon.get_function(gParameters['loss'])())
    optimizer = p1_common_neon.build_optimizer(gParameters['optimizer'],
                                            gParameters['learning_rate'],
                                            kerasDefaults)

    callbacks = Callbacks(mlp, eval_set=val, metric=Accuracy(), eval_freq = 1)

    # Seed random generator for training
    np.random.seed(seed)
    
    mlp.fit(train, optimizer=optimizer, num_epochs=gParameters['epochs'], cost=cost, callbacks=callbacks)

    # model save
    #save_fname = "model_mlp_W_" + ext
    #mlp.save_params(save_fname)

    # Evalute model on test set
    print('Model evaluation by neon: ', mlp.eval(test, metric=Accuracy()))
    y_pred = mlp.get_outputs(test)
    #print ("Shape y_pred: ", y_pred.shape)
    scores = p1b2.evaluate_accuracy(p1_common.convert_to_class(y_pred), y_test)
    print('Evaluation on test data:', scores)


if __name__ == '__main__':
    main()





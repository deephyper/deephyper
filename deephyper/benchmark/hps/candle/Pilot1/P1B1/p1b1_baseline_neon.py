from __future__ import division, print_function

import os
import sys
import argparse
import logging


import numpy as np

# For non-interactive plot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import neon
from neon.util.argparser import NeonArgparser
from neon.data import ArrayIterator
from neon.callbacks.callbacks import Callbacks
from neon.layers import GeneralizedCost, Affine, Dropout, Reshape
from neon.models import Model
from neon.backends import gen_backend

#from neon import logger as neon_logger

import p1b1
import p1_common
import p1_common_neon


def get_p1b1_parser():
    
    # Construct neon arg parser. It generates a large set of options by default
    parser = NeonArgparser(__doc__)
    # Specify the default config_file
    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(p1b1.file_path, 'p1b1_default_model.txt'),
                        help="specify model configuration file")
    
    # Parse other options that are not included on neon arg parser
    parser = p1_common.get_p1_common_parser(parser)
    
    
    return parser
    
    
def main():
    # Get command-line parameters
    parser = get_p1b1_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b1.read_config_file(args.config_file)
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
    ext = p1b1.extension_from_parameters(gParameters, '.neon')
    
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

    input_dim = X_train.shape[1]
    output_dim = input_dim

    # Re-generate the backend after consolidating parsing and file config
    gen_backend(backend=args.backend,
                rng_seed=seed,
                device_id=args.device_id,
                batch_size=gParameters['batch_size'],
                datatype=gParameters['datatype'],
                max_devices=args.max_devices,
                compat_mode=args.compat_mode)

    # Set input and target to X_train
    train = ArrayIterator(X_train)
    val = ArrayIterator(X_val)
    test = ArrayIterator(X_test)

    # Initialize weights and learning rule
    initializer_weights = p1_common_neon.build_initializer(gParameters['initialization'], kerasDefaults)
    initializer_bias = p1_common_neon.build_initializer('constant', kerasDefaults, 0.)

    activation = p1_common_neon.get_function(gParameters['activation'])()

    # Define Autoencoder architecture
    layers = []
    reshape = None

    # Autoencoder
    layers_params = gParameters['dense']
    
    if layers_params != None:
        if type(layers_params) != list:
            layers_params = list(layers_params)
        # Encoder Part
        for i,l in enumerate(layers_params):
            layers.append(Affine(nout=l, init=initializer_weights, bias=initializer_bias, activation=activation))
        # Decoder Part
        for i,l in reversed( list(enumerate(layers_params)) ):
            if i < len(layers)-1:
                layers.append(Affine(nout=l, init=initializer_weights, bias=initializer_bias, activation=activation))

    layers.append(Affine(nout=output_dim, init=initializer_weights, bias=initializer_bias, activation=activation))
    
    # Build Autoencoder model
    ae = Model(layers=layers)

    # Define cost and optimizer
    cost = GeneralizedCost(p1_common_neon.get_function(gParameters['loss'])())
    optimizer = p1_common_neon.build_optimizer(gParameters['optimizer'],
                                            gParameters['learning_rate'],
                                            kerasDefaults)

    callbacks = Callbacks(ae, eval_set=val, eval_freq = 1)

    # Seed random generator for training
    np.random.seed(seed)


    ae.fit(train, optimizer=optimizer, num_epochs=gParameters['epochs'], cost=cost, callbacks=callbacks)

    # model save
    #save_fname = "model_ae_W" + ext
    #ae.save_params(save_fname)

    # Compute errors
    X_pred = ae.get_outputs(test)
    scores = p1b1.evaluate_autoencoder(X_pred, X_test)
    print('Evaluation on test data:', scores)

    diff = X_pred - X_test
    # Plot histogram of errors comparing input and output of autoencoder
    plt.hist(diff.ravel(), bins='auto')
    plt.title("Histogram of Errors with 'auto' bins")
    plt.savefig('histogram_neon.png')



if __name__ == '__main__':
    main()





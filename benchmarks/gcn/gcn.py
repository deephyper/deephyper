from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import time
import os
import sys
from pprint import pprint
import hashlib
import pickle
from keras.models import load_model

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
from deephyper.benchmarks import keras_cmdline 

BNAME = 'GCN'

def extension_from_parameters(param_dict):
    extension = ''
    for key in sorted(param_dict):
        if key != 'epochs':
            print ('%s: %s' % (key, param_dict[key]))
            extension += '.{}={}'.format(key,param_dict[key])
    print(extension)
    return extension

def save_meta_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_meta_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data


def defaults():
    def_parser = keras_cmdline.create_parser()
    def_parser = augment_parser(def_parser)
    return vars(def_parser.parse_args(''))

def run(param_dict):
    default_params = defaults()
    for key in default_params:
        if key not in param_dict:
            param_dict[key] = default_params[key]
    pprint(param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)
    print(param_dict)

    EPOCHS = param_dict['epochs']
    FILTER = param_dict['filter']
    MAX_DEGREE = param_dict['max_degree']
    SYM_NORM = param_dict['sys_norm']
    DROPOUT = param_dict['dropout']
    NUNITS = param_dict['nunits']
    CHECKPOINT = param_dict['checkpoint']
    ACTIVATION = param_dict['activation']
    #SHARE_WEIGHTS = param_dict['share_weights']
    # Define parameters
    DATASET = 'cora'
    #FILTER = 'localpool'  # 'chebyshev'
    #MAX_DEGREE = 2  # maximum polynomial degree
    #SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    

    PATIENCE = 10  # early stopping patience

    # Get data
    X, A, y = load_data(dataset=DATASET)
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    # Normalize X
    X /= X.sum(1).reshape(-1, 1)

    extension = extension_from_parameters(param_dict)
    hex_name = hashlib.sha224(extension.encode('utf-8')).hexdigest()
    model_name = '{}-{}.h5'.format(BNAME, hex_name)
    model_mda_name = '{}-{}.pkl'.format(BNAME, hex_name)
    initial_epoch = 0

    resume = False

    if os.path.exists(model_name) and os.path.exists(model_mda_name):
        print('model and meta data exists; loading model from h5 file')
        model = load_model(model_name, custom_objects={'GraphConvolution': GraphConvolution})
        saved_param_dict = load_meta_data(model_mda_name)
        initial_epoch = saved_param_dict['epochs']
        if initial_epoch < param_dict['epochs']:
            resume = True
        else:
            initial_epoch = 0

    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        A_ = preprocess_adj(A, SYM_NORM)
        support = 1
        graph = [X, A_]
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        support = MAX_DEGREE + 1
        graph = [X]+T_k
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    else:
        raise Exception('Invalid filter type.')

    if not resume:
        X_in = Input(shape=(X.shape[1],))        
        # Define model architecture
        # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
        # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
        H = Dropout(DROPOUT)(X_in)
        H = GraphConvolution(NUNITS, support, activation=ACTIVATION, kernel_regularizer=l2(5e-4))([H]+G)
        H = Dropout(DROPOUT)(H)
        Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

        # Compile model
        model = Model(inputs=[X_in]+G, outputs=Y)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # Helper variables for main training loop
        wait = 0
        preds = None
        best_val_loss = 99999

    # Fit
    for epoch in range(initial_epoch, EPOCHS):
        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                    [idx_train, idx_val])
        print("Epoch: {:04d}".format(epoch),
            "train_loss= {:.4f}".format(train_val_loss[0]),
            "train_acc= {:.4f}".format(train_val_acc[0]),
            "val_loss= {:.4f}".format(train_val_loss[1]),
            "val_acc= {:.4f}".format(train_val_acc[1]),
            "time= {:.4f}".format(time.time() - t))

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
        "loss= {:.4f}".format(test_loss[0]),
        "accuracy= {:.4f}".format(test_acc[0]))
    print('===Validation accuracy:', train_val_acc[1])
    print('OUTPUT:', -train_val_acc[1])
    
    if CHECKPOINT:
        model.save(model_name)  
        save_meta_data(param_dict, model_mda_name)

    return -train_val_acc[1]

def augment_parser(parser):

    parser.add_argument('--sys_norm', action='store', dest='sys_norm',
                        nargs='?', const=1, type=bool, default=False,
                        help='boolean. Whether to apply sys norm?')

    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='2',
                        help='number of units in graph convolution layers')

    parser.add_argument('--filter', action='store',
                        dest='filter',
                        nargs='?', const=1, type=str, default='localpool',
                        choices=['localpool', 'chebyshev'],
                        help='type of filter')

    parser.add_argument('--max_degree', action='store', dest='max_degree',
                        nargs='?', const=2, type=int, default='2',
                        help='maximum degree')

    parser.add_argument('--checkpoint', action='store', dest='checkpoint',
                        nargs='?', const=1, type=bool, default=False,
                        help='boolean. Whether to save model?')

    return parser

if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    print(param_dict)
    run(param_dict)

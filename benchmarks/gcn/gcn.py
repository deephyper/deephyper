import sys
import os
from pprint import pprint
import  time
here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

from deephyper.benchmarks import util

timer = util.Timer()
timer.start('module loading')
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import hashlib
import pickle
from keras.models import load_model

from deephyper.benchmarks import keras_cmdline 
from keras.callbacks import EarlyStopping
from deephyper.benchmarks.util import TerminateOnTimeOut
from numpy.random import seed
from tensorflow import set_random_seed
timer.end()

seed(1)
set_random_seed(2)


def run(param_dict):
    param_dict = keras_cmdline.fill_missing_defaults(augment_parser, param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)
    pprint(param_dict)

    EPOCHS = param_dict['epochs']
    FILTER = param_dict['filter']
    MAX_DEGREE = param_dict['max_degree']
    SYM_NORM = param_dict['sys_norm']
    DROPOUT = param_dict['dropout']
    NUNITS = param_dict['nunits']
    ACTIVATION = param_dict['activation']
    BATCH_SIZE = param_dict['batch_size']
    TIMEOUT = param_dict['timeout']

    #SHARE_WEIGHTS = param_dict['share_weights']
    # Define parameters
    DATASET = 'cora'
    #FILTER = 'localpool'  # 'chebyshev'
    #MAX_DEGREE = 2  # maximum polynomial degree
    #SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    
    PATIENCE = 10  # early stopping patience

    # Get data
    timer.start('stage in')
    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data/cora')

    paths = util.stage_in(['cora.content', 'cora.cites'],
                          source=data_source,
                          dest=param_dict['stage_in_destination'])
    path_content = paths['cora.content']
    path_cites = paths['cora.cites']

    idx_features_labels = np.genfromtxt(path_content, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path_cites, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(
        adj.shape[0], edges.shape[0], features.shape[1]))
    X, A, y = features.todense(), adj, labels
    timer.end()

    timer.start('preprocessing')
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    # Normalize X
    X /= X.sum(1).reshape(-1, 1)

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

    model_path = param_dict['model_path']
    model_mda_path = None
    model = None
    initial_epoch = 0

    if model_path:
        custom_objects = {'GraphConvolution': GraphConvolution}
        savedModel = util.resume_from_disk(BNAME, param_dict,
                data_dir=model_path, custom_objects=custom_objects)
        model_mda_path = savedModel.model_mda_path
        model_path = savedModel.model_path
        model = savedModel.model
        initial_epoch = savedModel.initial_epoch

    if model is None:
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
    timer.end()
    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='auto')
    timeout_monitor = TerminateOnTimeOut(TIMEOUT)
    callbacks_list = [timeout_monitor]
    # Fit
    training_timer = util.Timer()
    training_timer.start('model training')
    prev_val_acc = 0
    count = 0
    patience = 50
    delta = 0.0001
    for epoch in range(initial_epoch, EPOCHS):
        # Log wall-clock time
        timer.start(f'epoch {epoch}')
        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0]) #

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
        print("Epoch: {:04d}".format(epoch),
            "train_loss= {:.4f}".format(train_val_loss[0]),
            "train_acc= {:.4f}".format(train_val_acc[0]),
            "val_loss= {:.4f}".format(train_val_loss[1]),
            "val_acc= {:.4f}".format(train_val_acc[1]))
        timer.end()

        diff = abs(prev_val_acc - train_val_acc[1])
        #print(diff)
        if diff > delta:
            prev_val_acc = train_val_acc[1]
            count = 0
        else:
            count = count+1
        
        if count >= patience:
            print('Early stopping')
            break

        elapsed = time.time() - training_timer.t0
        if elapsed >= TIMEOUT * 60:
            print(' - timeout: training time = %2.3fs/%2.3fs' % (elapsed, TIMEOUT * 60))
            break
    training_timer.end()

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
        "loss= {:.4f}".format(test_loss[0]),
        "accuracy= {:.4f}".format(test_acc[0]))
    print('===Validation accuracy:', test_acc[0])
    print('OUTPUT:', -test_acc[0])
    
    if model_path:
        timer.start('model save')
        model.save(model_path)  
        util.save_meta_data(param_dict, model_mda_path)
        timer.end()

    return -test_acc[0]

def augment_parser(parser):

    parser.add_argument('--sys_norm', action='store', dest='sys_norm',
                        nargs='?', const=1, type=util.str2bool, default=False,
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

    return parser

if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    print(param_dict)
    run(param_dict)

#%%
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import numpy as np
# from Data_Processing import molecule
# from train_utils import load_data_split, load_dict, load_data_func
from spektral.datasets import citation
from spektral.utils import localpooling_filter
from sklearn.model_selection import train_test_split
from spektral.datasets import qm9
from spektral.utils import label_to_one_hot

def load_data():
    # Load data
    A, X, E, y = qm9.load_data(return_type='numpy',
                               nf_keys='atomic_num',
                               ef_keys='type',
                               self_loops=True,
                               amount=1000)  # Set to None to train on whole dataset
    y = y[['cv']].values  # Heat capacity at 298.15K

    # Preprocessing
    uniq_X = np.unique(X)
    uniq_X = uniq_X[uniq_X != 0]
    X = label_to_one_hot(X, uniq_X)
    uniq_E = np.unique(E)
    uniq_E = uniq_E[uniq_E != 0]
    E = label_to_one_hot(E, uniq_E)

    # Parameters
    N = X.shape[-2]  # Number of nodes in the graphs
    F = X.shape[-1]  # Node features dimensionality
    S = E.shape[-1]  # Edge features dimensionality
    n_out = y.shape[-1]  # Dimensionality of the target
    learning_rate = 1e-3  # Learning rate for SGD
    epochs = 25  # Number of training epochs
    batch_size = 32  # Batch size
    es_patience = 5  # Patience fot early stopping

    # Train/test split
    A_train, A_test, \
    X_train, X_test, \
    E_train, E_test, \
    y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)
    print(f"================ Load Data ================")
    print(f"A_train shape: {np.shape(A_train)}")
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"E_train shape: {np.shape(E_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"A_test shape: {np.shape(A_test)}")
    print(f"X_test shape: {np.shape(X_test)}")
    print(f"E_test shape: {np.shape(E_test)}")
    print(f"y_test shape: {np.shape(y_test)}")
    return ([X_train, A_train, E_train], y_train), ([X_test, A_test, E_test], y_test)

#%%

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import collections

import tensorflow as tf

from deephyper.search.nas.model.space import AutoKSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.gnn import GraphConv2, EdgeConditionedConv2


def add_gcn_to_(node):
    activations = [tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for channels in [16]:
        for activation in activations:
            node.add_op(EdgeConditionedConv2(channels=channels, activation=activation, use_bias=False))
    return


def create_search_space(input_shape=None,
                        output_shape=None,
                        num_layers=1,
                        *args, **kwargs):
    if output_shape is None:
        output_shape = (1,)
    if input_shape is None:
        input_shape = [(8, 4), (8, 8), (8, 8, 3)]
    arch = AutoKSearchSpace(input_shape, output_shape, regression=True)
    prev_input = arch.input_nodes[0]
    prev_input1 = arch.input_nodes[1]
    prev_input2 = arch.input_nodes[2]

    vnode = VariableNode()
    add_gcn_to_(vnode)
    arch.connect(prev_input, vnode)
    arch.connect(prev_input1, vnode)
    arch.connect(prev_input2, vnode)
    # * Cell output
    cell_output = vnode

    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))
    prev_input = cmerge

    vnode = VariableNode()
    add_gcn_to_(vnode)
    arch.connect(prev_input, vnode)
    arch.connect(prev_input1, vnode)
    arch.connect(prev_input2, vnode)
    cmerge = ConstantNode()
    cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))
    prev_input = cmerge

    return arch

#%%


from deephyper.problem import NaProblem


Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space)  # , num_layers=3)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=10,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_r2', mode='max', verbose=0, patience=5  # or 'val_acc' ?
        )
    ),
)

Problem.loss("mse")  # or 'categorical_crossentropy' ?

Problem.metrics(['r2'])  # or 'acc' ?

Problem.objective('val_r2__last')  # or 'val_acc__last' ?

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

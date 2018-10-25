from collections import OrderedDict

from nas.contrib.anl_mlp_2 import create_structure
from nas.model.preprocessing import stdscaler_minmax
from deephyper.benchmarks.mnist1DNas.load_data import load_data
from deephyper.benchmarks import Problem

Problem = Problem()
Problem.add_dim('regression', False)
Problem.add_dim('load_data', {
    'func': load_data
})
Problem.add_dim('create_structure', {
    'func': create_structure,
    'kwargs': {
        'num_cells': 5
    }
})
Problem.add_dim('hyperparameters', {
    'batch_size': 100,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'num_epochs': 50,
    'loss_metric': 'mean_softmax_cross_entropy',
    'test_metric': 'accuracy'
})


if __name__ == '__main__':
    print(Problem)

# TODO : not ready

from deephyper.searches.nas.contrib.google_nas_net import create_structure
from deephyper.benchmarks.nas.mnist2D.load_data import load_data
from deephyper.benchmarks import Problem

Problem = Problem()
Problem.add_dim('regression', False)
Problem.add_dim('load_data', {
    'func': load_data
})
Problem.add_dim('create_structure', {
    'func': create_structure,
    'kwargs': {}
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

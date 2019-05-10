from deephyper.search.nas.model.baseline.anl_mlp_2 import create_structure
from deephyper.benchmark.nas.mnist1D.load_data import load_data
from deephyper.benchmark import Problem

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
    'batch_size': 64,
    'learning_rate': 0.0001,
    'optimizer': 'adam',
    'num_epochs': 10,
    'loss_metric': 'categorical_crossentropy',
    'metrics': ['acc']
})


if __name__ == '__main__':
    print(Problem)

''' * @Author: romain.egele, dipendra.jha * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

from nas.cell.structure import create_sequential_structure
from nas.cell.mlp import create_dense_cell_example
from deephyper.benchmarks.mnistNas import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['num_outputs'] = 10
        space['regression'] = False
        space['load_data'] = {
            'func': load_data
        }

        # ARCH
        space['num_cells'] = 2
        space['create_structure'] = {
            'func': create_sequential_structure,
            'kwargs': {
                'num_cells': 2
            }
        }
        space['create_cell'] = {
            'func': create_dense_cell_example
        }

        # HyperParameters
        space['hyperparameters'] = {
            'batch_size': 64,
            'eval_batch_size': 64,
            'activation': 'relu',
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'num_epochs': 1,
            'loss_metric': 'softmax_cross_entropy',
            'test_metric': 'accuracy',
        }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

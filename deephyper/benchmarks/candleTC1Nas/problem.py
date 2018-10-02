''' * @Author: romain.egele * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

from nas.cell.structure import create_sequential_structure
from nas.cell.mlp import create_dense_cell_type1
from deephyper.benchmarks.candleTC1Nas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['regression'] = True
        space['load_data'] = {
            'func': load_data
        }

        # ARCH
        space['create_structure'] = {
            'func': create_sequential_structure,
            'kwargs': {
                'num_cells': 3
            }
        }
        space['create_cell'] = {
            'func': create_dense_cell_type1
        }

        # HyperParameters
        space['hyperparameters'] = {
            'batch_size': 64,
            'eval_batch_size': 64,
            'activation': 'relu',
            'learning_rate': 0.1,
            'optimizer': 'adam',
            'num_epochs': 20,
            'loss_metric': 'mean_squared_error',
            'test_metric': 'mean_squared_error',
        }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

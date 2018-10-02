''' * @Author: romain.egele * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

from nas.cell.structure import create_sequential_structure
from nas.cell.mlp import create_dense_cell_type1
from deephyper.benchmarks.candleP1B3Nas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['input_shape'] = [29532]
        space['output_shape'] = [1]
        space['regression'] = True
        space['load_data'] = {
            'func': load_data,
            'kwargs' : {
                'batch_size': 100,
            }
        }

        # ARCH
        space['create_structure'] = {
            'func': create_sequential_structure,
            'kwargs': {
                'num_cells': 1
            }
        }
        space['create_cell'] = {
            'func': create_dense_cell_type1
        }

        # HyperParameters
        space['hyperparameters'] = {
            'eval_freq': 100,
            'batch_size': 100,
            'learning_rate': 0.1,
            'optimizer': 'adam',
            'num_epochs': 20,
            'loss_metric': 'mean_squared_error',
            'test_metric': 'mean_squared_error',
        }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

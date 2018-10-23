''' * @Author: romain.egele * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

from nas.contrib.anl_mlp_2 import create_structure
from nas.model.preprocessing import stdscaler_minmax
from deephyper.benchmarks.ackleyRegNas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['regression'] = True

        space['load_data'] = {
            'func': load_data
        }

        space['preprocessing'] = {
            'func': stdscaler_minmax
        }

        # ARCH
        space['create_structure'] = {
            'func': create_structure,
            'kwargs': {
                'num_cells': 5
            }
        }

        # HyperParameters
        space['hyperparameters'] = {
            'batch_size': 64,
            'learning_rate': 0.1,
            'optimizer': 'adam',
            'num_epochs': 20,
            'loss_metric': 'mean_squared_error',
            'test_metric': 'mean_squared_error',
        }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

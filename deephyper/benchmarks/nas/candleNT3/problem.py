from collections import OrderedDict
#from nas.contrib.anl_mlp_1 import create_structure
from nas.contrib.anl_mlp_2 import create_structure
from deephyper.benchmark.candleNT3Nas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['regression'] = True
        space['load_data'] = {
            'func': load_data
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
            'batch_size': 32,
            'activation': 'relu',
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'num_epochs': 10,
            'loss_metric': 'mean_squared_error',
            'test_metric': 'mean_squared_error',
        }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

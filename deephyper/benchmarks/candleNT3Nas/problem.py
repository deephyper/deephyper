'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict
from nas.contrib.anl_mlp_1 import create_structure
from deephyper.benchmarks.candleNT3Nas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['regression'] = False
        space['load_data'] = {
            'func': load_data
        }

        # ARCH
        space['create_structure'] = {
            'func': create_structure,
            'kwargs': {
                'num_cells': 4
            }
        }

        # HyperParameters
        space['hyperparameters'] = {'batch_size': 64,
                                    'eval_batch_size': 64,#needs to be same as batch size
                                    'activation': 'relu',
                                    'learning_rate': 0.001,
                                    'optimizer': 'adam',
                                    'num_epochs': 10,
                                    'loss_metric': 'softmax_cross_entropy',
                                    'test_metric': 'accuracy',
                                    'eval_freq': 1
                                }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

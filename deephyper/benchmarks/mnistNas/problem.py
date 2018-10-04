''' * @Author: romain.egele, dipendra.jha * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

from nas.contrib.google_nas_net import create_structure
from deephyper.benchmarks.mnistNas.load_data import load_data

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['num_outputs'] = 10
        space['regression'] = False
        space['load_data'] = {
            'func': load_data
        }

        # ARCH
        space['create_structure'] = {
            'func': create_structure,
            'kwargs': {
                'n_normal': 2
            }
        }

        # HyperParameters
        space['hyperparameters'] = {
            'batch_size': 64,
            'eval_batch_size': 64, # TODO : check if useful
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

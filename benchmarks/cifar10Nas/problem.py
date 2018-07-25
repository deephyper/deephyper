'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict
from deephyper.model.arch import StateSpace

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['num_outputs'] = 10
        space['regression'] = False

        # ARCH
        space['max_layers'] = 5
        space['layer_type'] = 'conv2D'
        state_space = StateSpace()
        state_space.add_state('filter_height', [size for size in range(3,6,2)])
        state_space.add_state('filter_width', [size for size in range(3,6,2)])
        state_space.add_state('pool_height', [size for size in range(1,3)])
        state_space.add_state('pool_width', [size for size in range(1,3)])
        state_space.add_state('stride_height', [s for s in range(1,2)])
        state_space.add_state('stride_width', [s for s in range(1,2)])
        state_space.add_state('drop_out', [])
        state_space.add_state('num_filters', [2**i for i in range(5, 10)])
        state_space.add_state('skip_conn', [])

        space['state_space'] = state_space

        # HyperParameters
        space['hyperparameters'] = {'batch_size': 64,
                                    'eval_batch_size': 32,
                                    'activation': 'relu',
                                    'learning_rate': 0.001,
                                    'optimizer': 'adam',
                                    'num_epochs': 50,
                                    'loss_metric': 'softmax_cross_entropy',
                                    'test_metric': 'accuracy',
                                    'eval_freq': 100
                                }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

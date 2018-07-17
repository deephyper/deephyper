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
        space['max_layers'] = 3
        space['layer_type'] = 'conv2D'
        #space['features'] = ['num_filters', 'filter_width', 'filter_height', 'pool_width', 'pool_height', 'stride_width', 'stride_height', 'drop_out']
        state_space = StateSpace()
        state_space.add_state('filter_height', [size for size in range(5,30,2)])
        state_space.add_state('filter_width', [size for size in range(5,30,2)])
        state_space.add_state('pool_height', [size for size in range(5,30,2)])
        state_space.add_state('pool_width', [size for size in range(5,30,2)])
        state_space.add_state('stride_height', [s for s in range(2,20,2)])
        state_space.add_state('stride_width', [s for s in range(2,20,2)])
        state_space.add_state('drop_out', [])
        state_space.add_state('num_filters', [n for n in range(5,35,5)])
        state_space.add_state('skip_conn', [])
        space['state_space'] = state_space

        # ITER
        space['max_episodes'] = 10 # iter on controller

        # HyperParameters
        space['hyperparameters'] = {'batch_size': 32,
                                    'eval_batch_size': 32,
                                    'activation': 'relu',
                                    'learning_rate': 0.001,
                                    'optimizer': 'adam',
                                    'num_epochs': 10,
                                    'loss_metric': 'softmax_cross_entropy',
                                    'test_metric': 'accuracy',
                                    'eval_freq': 100
                                }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

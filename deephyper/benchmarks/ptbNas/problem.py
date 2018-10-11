'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict
from deephyper.model.arch import StateSpace

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['num_outputs'] = 1
        space['unit_type'] = 'LSTM'
        space['num_features'] = 1
        space['regression'] = True
        space['text_input'] = True
        space['num_units'] = 20

        # ARCH
        space['max_layers'] = 2
        space['layer_type'] = 'rnn'
        state_space = StateSpace()
        state_space.add_state('num_units', [2 ** i for i in range(5, 6)])
        state_space.add_state('drop_out', [])

        space['state_space'] = state_space

        # ITER
        space['max_episodes'] = 50 # iter on controller

        # HyperParameters
        space['hyperparameters'] = {'batch_size': 20,
                                    'activation': 'relu',
                                    'learning_rate': 1.0,
                                    'max_grad_norm':20,
                                    'optimizer': 'adam',
                                    'num_epochs': 55,
                                    'loss_metric': 'sequence_loss_by_example',
                                    'test_metric': 'perplexity'
                                }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

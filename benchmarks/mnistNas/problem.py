'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-21 15:31:30
'''

from collections import OrderedDict

class Problem:
    def __init__(self):
        space = OrderedDict()

        # I/O
        space['input_shape'] = [28,28, 1]

        space['num_outputs'] = 10

        # ARCH
        space['max_layers'] = 2
        space['layer_type'] = 'conv2D'
        space['features'] = ['num_filters', 'filter_width', 'filter_height', 'pool_width',
            'pool_height', 'stride_width', 'stride_height', 'drop_out']

        # ITER
        space['max_episodes'] = 2500 # iter on controller

        # HyperParameters
        space['hyperparameters'] = { 'batch_size': 32,
                                   'activation': 'relu',
                                   'learning_rate': 0.001,
                                   'optimizer': 'adam',
                                   'num_epochs': 10,
                                   'loss_metric': 'softmax_cross_entropy',
                                   'test_metric': 'accuracy'
                                }
        self.space = space
        self.params = self.space.keys()
        self.starting_point = [ 1.0 for i in range(space['max_layers']*len(space['features']))]

        # space['data'] = ...


if __name__ == '__main__':
    instance = Problem()

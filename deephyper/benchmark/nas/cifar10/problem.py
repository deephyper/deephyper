# TODO : not ready

from collections import OrderedDict
# OUTDATED

class Problem:
    def __init__(self):
        space = OrderedDict()
        space['regression'] = False

        # TODO
        raise NotImplementedError

        # HyperParameters
        space['hyperparameters'] = {'batch_size': 64,
                                    'activation': 'relu',
                                    'learning_rate': 0.1,
                                    'optimizer': 'momentum',
                                    'num_epochs': 50,
                                    'loss_metric': 'softmax_cross_entropy',
                                    'test_metric': 'accuracy',
                                }
        self.space = space


if __name__ == '__main__':
    instance = Problem()

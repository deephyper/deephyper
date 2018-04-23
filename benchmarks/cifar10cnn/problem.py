from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (10, 100)
        #bechmark specific parameters
        space['activation'] = ['softmax', 'relu', 'sigmoid']
        space['f1_size'] = [1, 3, 5]
        space['f2_size'] = [1, 3, 5]
        space['f1_units'] = [8, 16, 32, 64]
        space['f2_units'] = [8, 16, 32, 64]
        space['p_size'] = [2, 4, 6, 8]
        space['nunits'] = (100, 1000)
        #network parameters
        space['batch_size'] = (8, 1024)
        space['dropout'] = (0.0, 1.0)
        space['dropout2'] = (0.0, 1.0)
        space['optimizer'] = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
        # common optimizer parameters
        #space['clipnorm'] = (1e-04, 1e01)
        #space['clipvalue'] = (1e-04, 1e01)
        # optimizer parameters
        space['learning_rate'] = (1e-04, 1e01)
        #space['momentum'] =  (0, 1e01)
        #space['decay'] =  (0, 1e01)
        #space['nesterov'] = [False, True]
        #space['rho'] = (1e-04, 1e01)
        #space['epsilon'] = (1e-08, 1e01)
        #space['beta1'] = (1e-04, 1e01)
        #space['beta2'] = (1e-04, 1e01)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [10, 'relu', 3, 3, 32, 64, 2, 512, 32, 0.25, 0.5,
                               'rmsprop', 0.001]


if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))

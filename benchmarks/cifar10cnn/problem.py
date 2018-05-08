from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (5, 500)
        #bechmark specific parameters
        space['f1_size'] = [1, 3, 5, 7]
        space['f2_size'] = [1, 3, 5, 7]
        space['f1_units'] = [8, 16, 32, 64]
        space['f2_units'] = [8, 16, 32, 64]
        space['p_size'] = [2, 3, 4]
        space['nunits'] = (1, 1000)
        #network parameters
        space['activation'] = ['relu', 'elu', 'selu', 'tanh'] #['relu', 'softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid']
        space['batch_size'] = (8, 1024)
        space['dropout'] = (0.0, 1.0)
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
        self.starting_point = [5, 1, 1, 8, 8, 1, 1, 'relu', 8, 0.0, 'sgd', 1e-04]

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))

from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (5, 500)
        #bechmark specific parameters
        space['rnn_type'] = ['LSTM', 'GRU', 'SimpleRNN']
        space['embed_hidden_size'] = (1, 100)
        space['sent_hidden_size'] = (1, 100)
        space['query_hidden_size'] = (1, 100)
        #network parameters
        space['activation'] = ['relu', 'elu', 'selu', 'tanh']
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
        self.starting_point = [5, 'LSTM', 1, 1, 1, 'relu', 8, 0.0, 'sgd', 1e-04]

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))

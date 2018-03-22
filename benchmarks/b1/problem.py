from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (2, 500)
        #bechmark specific parameters
        space['rnn_type'] = ['LSTM', 'GRU', 'SimpleRNN']
        space['hidden_size'] = (10, 100)
        space['layers'] = (1, 30)
        #network parameters
        space['activation'] = ['softmax', 'elu', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid']
        #space['loss'] = ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'categorical_hinge', 'hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
        space['batch_size'] = (8, 1024)
        #space['init'] = ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform']
        #space['dropout'] = (0.0, 1.0)
        space['optimizer'] = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
        # common optimizer parameters
        space['clipnorm'] = (1e-04, 1e01)
        space['clipvalue'] = (1e-04, 1e01)
        # optimizer parameters
        space['learning_rate'] = (1e-04, 1e01)
        space['momentum'] =  (0, 1e01)
        space['decay'] =  (0, 1e01)
        space['nesterov'] = [False, True]
        space['rho'] = (1e-04, 1e01)
        space['epsilon'] = (1e-08, 1e01)
        space['beta1'] = (1e-04, 1e01)
        space['beta2'] = (1e-04, 1e01)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [200, 'LSTM', 10, 1, 'softmax', 5, 32, 'sgd', 1.0, 0.5, 0.01, 0, 0, False, 0.9, 1e-08, 0.9, 0.999]

if __name__ == '__main__':
    instance = Problem()
    print(instance.space)
    print(instance.params)

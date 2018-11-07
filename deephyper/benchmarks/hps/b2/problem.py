from deephyper.benchmarks import HpProblem

Problem = HpProblem()
Problem.add_dim('epochs', (5,30), default=5)
Problem.add_dim('rnn_type', ['LSTM', 'GRU', 'SimpleRNN'], default='LSTM')
Problem.add_dim('nhidden', (1, 100), default=1)

#network parameters
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'], default='relu')
Problem.add_dim('batch_size', (8, 1024), default=8)
Problem.add_dim('dropout', (0.0, 1.0), default=0.0)
Problem.add_dim('optimizer', ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], default='sgd')

# common optimizer parameters
Problem.add_dim('learning_rate', (1e-04, 1e01), default=1e-4)

if __name__ == '__main__':
    print(Problem)

from deephyper.benchmarks import HpProblem

Problem = HpProblem()
Problem.add_dim('epochs', (5, 500), 5)

# benchmark specific parameters
Problem.add_dim('nhidden', (1, 100), 1)
Problem.add_dim('nunits', (1, 1000), 1)

# network parameters
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'], 'relu')
Problem.add_dim('batch_size', (8, 1024), 8)
Problem.add_dim('dropout', (0.0, 1.0), 0.0)
Problem.add_dim('optimizer', ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], 'sgd')

# common optimizer parameters
#Problem.add_dim(['clipnorm'] = (1e-04, 1e01)
#Problem.add_dim(['clipvalue'] = (1e-04, 1e01)
# optimizer parameters
Problem.add_dim('learning_rate', (1e-04, 1e01), 1e-04)
#Problem.add_dim(['momentum'] =  (0, 1e01)
#Problem.add_dim(['decay'] =  (0, 1e01)
#Problem.add_dim(['nesterov'] = [False, True]
#Problem.add_dim(['rho'] = (1e-04, 1e01)
#Problem.add_dim(['epsilon'] = (1e-08, 1e01)
#Problem.add_dim(['beta1'] = (1e-04, 1e01)
#Problem.add_dim(['beta2'] = (1e-04, 1e01)


if __name__ == '__main__':
    print(Problem)

from deephyper.benchmark import HpProblem

Problem = HpProblem()
Problem.add_dim('epochs', (5, 500), 5)
Problem.add_dim('nunits_l1', (1, 1000), 1)
Problem.add_dim('nunits_l2', (1, 1000), 1)
Problem.add_dim('activation_l1', ['relu', 'elu', 'selu', 'tanh'], 'relu')
Problem.add_dim('activation_l2', ['relu', 'elu', 'selu', 'tanh'], 'relu')
Problem.add_dim('batch_size', (8, 1024), 8)
Problem.add_dim('dropout_l1', (0.0, 1.0), 0.0)
Problem.add_dim('dropout_l2', (0.0, 1.0), 0.0)


if __name__ == '__main__':
    print(Problem)

from deephyper.benchmark import HpProblem

Problem = HpProblem()

Problem.add_dim('units', (1, 100))
Problem.add_dim('activation', [None, 'relu', 'sigmoid', 'tanh'])
Problem.add_dim('lr', (0.0001, 1.))

Problem.add_starting_point(
    units=10,
    activation=None,
    lr=0.01)

from deephyper.benchmark import HpProblem

Problem = HpProblem()

Problem.add_dim('units_l1', (1, 100))
Problem.add_dim('activation_l1', [None, 'relu', 'sigmoid', 'tanh'])
Problem.add_dim('dropout_l1', (0., 1.))
Problem.add_dim('units_l2', (1, 100))
Problem.add_dim('activation_l2', [None, 'relu', 'sigmoid', 'tanh'])
Problem.add_dim('dropout_l2', (0., 1.))
Problem.add_dim('batch_size', (32, 512))
Problem.add_dim('lr', (0.0001, 1.))

Problem.add_starting_point(
    units_l1=10,
    activation_l1=None,
    dropout_l1=0.1,
    units_l2=10,
    activation_l2=None,
    dropout_l2=0.1,
    batch_size=64,
    lr=0.01
)

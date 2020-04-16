from deephyper.problem import HpProblem

Problem = HpProblem()
Problem.add_dim("epochs", (5, 500))
Problem.add_dim("nunits_l1", (1, 1000))
Problem.add_dim("nunits_l2", (1, 1000))
Problem.add_dim("activation_l1", ["relu", "elu", "selu", "tanh"])
Problem.add_dim("activation_l2", ["relu", "elu", "selu", "tanh"])
Problem.add_dim("batch_size", (8, 1024))
Problem.add_dim("dropout_l1", (0.0, 1.0))
Problem.add_dim("dropout_l2", (0.0, 1.0))


Problem.add_starting_point(
    epochs=5,
    nunits_l1=1,
    nunits_l2=2,
    activation_l1="relu",
    activation_l2="relu",
    batch_size=8,
    dropout_l1=0.0,
    dropout_l2=0.0,
)


if __name__ == "__main__":
    print(Problem)

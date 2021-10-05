from deephyper.problem import HpProblem

Problem = HpProblem()
Problem.add_hyperparameter((5, 500), "epochs")
Problem.add_hyperparameter((1, 1000), "nunits_l1")
Problem.add_hyperparameter((1, 1000), "nunits_l2")
Problem.add_hyperparameter(["relu", "elu", "selu", "tanh"], "activation_l1")
Problem.add_hyperparameter(["relu", "elu", "selu", "tanh"], "activation_l2")
Problem.add_hyperparameter((8, 1024), "batch_size")
Problem.add_hyperparameter((0.0, 1.0), "dropout_l1")
Problem.add_hyperparameter((0.0, 1.0), "dropout_l2")


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

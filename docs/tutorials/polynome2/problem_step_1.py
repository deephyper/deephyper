from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter((1, 100), "units")
Problem.add_hyperparameter(["identity", "relu", "sigmoid", "tanh"], "activation")
Problem.add_hyperparameter((0.0001, 1.0), "lr")

Problem.add_starting_point(units=10, activation="identity", lr=0.01)

if __name__ == "__main__":
    print(Problem)

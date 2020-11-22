from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter((10, 300), "n_estimators")
Problem.add_hyperparameter(["gini", "entropy"], "criterion")
Problem.add_hyperparameter((1, 50), "max_depth")
Problem.add_hyperparameter((2, 10), "min_samples_split")

# We define a starting point with the defaul hyperparameters from sklearn-learn
# that we consider good in average.
Problem.add_starting_point(
    n_estimators=100, criterion="gini", max_depth=50, min_samples_split=2
)

if __name__ == "__main__":
    print(Problem)
